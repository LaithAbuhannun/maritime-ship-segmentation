"""Explicit train/validation epochs for SAMSAR, Ship Contour and MMDetection."""
from pathlib import Path
import random
import numpy as np
from .tracking import EarlyStop
from .storage import atomic_json
from .data import evaluation_view
from .adapters import predict
from .reference.evaluate import coco_mask_eval

def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_state():
    import torch
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])

def save_checkpoint(path, state):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp'); torch.save(state,temp); temp.replace(path)

def ship_loss(model, batch, device):
    import torch
    from torch.nn import functional as F
    from .reference.models.ship_contour import _focal_loss
    batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    output = model(batch['image'], batch['initial'])
    mask = batch['valid'].unsqueeze(-1)
    gather = batch['indices'].unsqueeze(-1).expand(-1,-1,2)
    denominator = mask.sum().clamp_min(1)
    heatmap = _focal_loss(output['heatmap'],batch['heatmap'])
    size = (F.l1_loss(output['size'].flatten(2).transpose(1,2).gather(1,gather),
                     batch['sizes'],reduction='none')*mask).sum()/denominator
    offset = (F.l1_loss(output['offset'].flatten(2).transpose(1,2).gather(1,gather),
                       batch['offsets'],reduction='none')*mask).sum()/denominator
    contour = (F.smooth_l1_loss(output['contours'],batch['targets'],reduction='none')*
               mask.unsqueeze(-1)).sum()/denominator
    return {'loss':heatmap+size+offset+contour,'heatmap':heatmap,'size':size,'offset':offset,'contour':contour}, len(batch['image'])

def sam_loss(model, batch, device):
    import torch
    from .reference.models.paper_modules import samsar_hybrid_loss
    boxes = [v.to(device) for v in batch['boxes']]
    if not any(len(v) for v in boxes): return None, 0
    outputs = model(batch['images'].to(device),boxes)
    losses = []
    for output,truth in zip(outputs,batch['masks']):
        if not len(truth): continue
        logits = output['masks'][:,0]
        if logits.shape != truth.shape:
            raise ValueError(f'SAMSAR target/output mismatch: {logits.shape} vs {truth.shape}')
        losses.append(samsar_hybrid_loss(logits,truth.to(device)))
    if not losses: return None,0
    return {'loss':torch.stack(losses).mean()}, len(losses)

def mmdet_loss(model, batch, device):
    # training=True pads masks in the data preprocessor; model stays eval during validation.
    data = model.data_preprocessor(batch, training=True)
    losses = model(**data, mode='loss')
    terms = {k:sum(x.mean() for x in v) if isinstance(v,list) else v.mean()
             for k,v in losses.items() if 'loss' in k}
    return {'loss':sum(terms.values()), **terms}, len(data['data_samples'])

def setup(cfg, name, adapter, data):
    import torch
    from torch.utils.data import DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name in ('swininsseg','solov2_gala'):
        from .mmdetection import build_model_loaders
        train,val = build_model_loaders(cfg,name,adapter,data)
        return train,val,mmdet_loss,device
    if name == 'samsar':
        from .reference.models.samsar import (CocoPromptDataset,collate_prompts,
                                             _load_reference_builder,_freeze_sam_components)
        adapter.model = _load_reference_builder(cfg.samsar_repo)(sam_checkpoint=cfg.sam_checkpoint).to(device)
        _freeze_sam_components(adapter.model)
        dataset_cls,collate,size,loss = CocoPromptDataset,collate_prompts,cfg.samsar_img_size,sam_loss
    else:
        from .reference.models.ship_contour import ShipContourDataset,_build_network
        adapter.model = _build_network()().to(device)
        dataset_cls,collate,size,loss = ShipContourDataset,None,cfg.img_size,ship_loss
    options={}
    if name=='ship_contour':
        from collections import Counter
        from .storage import read_json
        maximum=max([32]+[max(Counter(a['image_id'] for a in read_json(data[f'coco_{s}'])['annotations']).values(),default=0)
                          for s in ('train','val')])
        options['max_instances']=maximum  # Never silently truncate dense training images.
    datasets = [dataset_cls(data[f'coco_{s}'],data[f'{s}_image_dir'],size,**options) for s in ('train','val')]
    loaders = [DataLoader(d,batch_size=cfg.batch_size if i==0 else 1,shuffle=i==0,
                         num_workers=cfg.workers,collate_fn=collate,persistent_workers=False)
               for i,d in enumerate(datasets)]
    return *loaders,loss,device

def loss_epoch(model, loader, objective, device, optimizer=None, scaler=None, amp=False):
    import torch
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            if training: optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp and device.type=='cuda'):
                losses,n = objective(model,batch,device)
            if losses is None: continue
            if not all(torch.isfinite(v).all() for v in losses.values()):
                raise ValueError('Non-finite loss; stop instead of saving misleading metrics')
            if training:
                scaler.scale(losses['loss']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                scaler.step(optimizer); scaler.update()
            for key,value in losses.items(): totals[key] = totals.get(key,0.)+float(value.detach())*n
            count += n
    if not count: raise ValueError('No usable samples in loss epoch')
    return {k:v/count for k,v in totals.items()}

def build_optimizer(cfg, name, model):
    """Use the optimizer family required by each reference architecture."""
    import torch
    parameters = [p for p in model.parameters() if p.requires_grad]
    if name == 'solov2_gala':
        # SOLOv2's MMDetection recipe is SGD-based.  The previous shared
        # AdamW optimizer left both SSDD and HRSID at zero validation AP.
        return torch.optim.SGD(parameters, lr=cfg.learning_rate,
                               momentum=.9, weight_decay=1e-4)
    return torch.optim.AdamW(parameters, lr=cfg.learning_rate)

def train(cfg,name,adapter,data,tracker,run_dir):
    import torch
    seed_all(cfg.seed)
    train_loader,val_loader,objective,device = setup(cfg,name,adapter,data)
    model = adapter.model
    optimizer = build_optimizer(cfg,name,model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg.max_epochs)
    enabled=cfg.amp and device.type=='cuda'
    scaler = (torch.amp.GradScaler('cuda',enabled=enabled) if hasattr(torch.amp,'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=enabled))
    stopper = EarlyStop(cfg.patience,cfg.min_delta)
    last,best = Path(run_dir)/'checkpoints/last.pt',Path(run_dir)/'checkpoints/best.pt'
    start = 0
    if last.exists():
        saved = torch.load(last,map_location='cpu',weights_only=False)
        if saved['split_fingerprint'] != data['fingerprint']: raise ValueError('Checkpoint split mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        stopper.load_state_dict(saved['early_stop']); restore_rng(saved['rng']); start=saved['epoch']
        if saved.get('history_row') and (not tracker.rows or tracker.rows[-1]['epoch'] < start):
            tracker.epoch(saved['history_row'])
    for epoch in range(start+1,cfg.max_epochs+1):
        if stopper.bad_epochs >= cfg.patience: break
        train_values = loss_epoch(model,train_loader,objective,device,optimizer,scaler,cfg.amp)
        val_values = loss_epoch(model,val_loader,objective,device,amp=cfg.amp)
        predictions = predict(adapter,evaluation_view(data,'val'))
        ap = coco_mask_eval(data['coco_val'],predictions,cfg.max_dets)
        row = {'epoch':epoch,**{'train_'+k:v for k,v in train_values.items()},
               **{'val_'+k:v for k,v in val_values.items()},
               **{'val_mask_'+k:v for k,v in ap.items() if isinstance(v,(float,int))},
               'lr':optimizer.param_groups[0]['lr']}
        improved,stop = stopper.update(row['val_mask_AP50_95'],epoch)
        scheduler.step()
        saved = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                 'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                 'rng':rng_state(),'early_stop':stopper.state_dict(),'epoch':epoch,
                 'split_fingerprint':data['fingerprint'],'history_row':row}
        if improved: save_checkpoint(best,saved)
        save_checkpoint(last,saved)
        tracker.epoch(row)
        if stop: break
    if not best.exists(): raise RuntimeError('No validation-selected checkpoint')
    atomic_json(Path(run_dir)/'training_complete.json',{'epochs':saved['epoch'],
        'best_epoch':stopper.best_epoch,'best_val_mask_AP50_95':stopper.best,
        'reason':'patience' if stopper.bad_epochs>=cfg.patience else 'max_epochs'})
    return best
