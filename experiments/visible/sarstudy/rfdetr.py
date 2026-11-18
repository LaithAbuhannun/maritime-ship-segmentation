"""RF-DETR Lightning integration; missing metrics fail rather than disappear."""
from pathlib import Path
import json
import shutil
from .tracking import EarlyStop
from .storage import atomic_json

def layout(data, destination):
    destination = Path(destination)
    for framework_split,split in (('train','train'),('valid','val')):
        folder = destination/framework_split; folder.mkdir(parents=True,exist_ok=True)
        payload = json.loads(Path(data[f'coco_{split}']).read_text())
        atomic_json(folder/'_annotations.coco.json',payload)
        for info in payload['images']:
            source = Path(data[f'{split}_image_dir'])/info['file_name']
            target = folder/info['file_name']
            if not target.exists():
                try: target.symlink_to(source.resolve())
                except OSError: shutil.copy2(source,target)
    # Test is deliberately absent: final testing uses the shared evaluator.
    return str(destination)

def train(cfg,name,adapter,data,tracker,run_dir):
    import torch
    from pytorch_lightning import Callback
    from rfdetr.config import RFDETRSegNanoConfig, SegmentationTrainConfig
    from rfdetr.training import RFDETRModelModule, RFDETRDataModule, build_trainer
    folder = Path(run_dir)/'checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    last,best = folder/'last.ckpt',folder/'best.pth'
    stopper = EarlyStop(cfg.patience,cfg.min_delta)
    class StudyCallback(Callback):
        resume_rng = None
        def state_dict(self): return {'stopper':stopper.state_dict()}
        def load_state_dict(self,state): stopper.load_state_dict(state['stopper'])
        def on_train_epoch_end(self,trainer,module):
            if trainer.sanity_checking: return
            metrics = {k:float(v.detach().cpu()) for k,v in trainer.callback_metrics.items()
                       if torch.is_tensor(v) and v.numel()==1}
            train_key = 'train/loss_epoch' if 'train/loss_epoch' in metrics else 'train/loss'
            required = [train_key,'val/loss','val/segm_mAP_50_95']
            missing = [k for k in required if k not in metrics]
            if missing: raise RuntimeError(f'RF-DETR required metrics absent: {missing}')
            row = {'epoch':trainer.current_epoch+1,'train_loss':metrics[train_key],
                   'val_loss':metrics['val/loss'],
                   'val_mask_AP50_95':100*metrics['val/segm_mAP_50_95'],
                   **{k.replace('/','_'):v for k,v in metrics.items()}}
            improved,stop = stopper.update(row['val_mask_AP50_95'],row['epoch'])
            if improved:
                from .native import save_checkpoint
                save_checkpoint(best,{'model':module.model.state_dict(),
                                     'epoch':row['epoch'],'num_classes':1})
            tracker.epoch(row)
            trainer.should_stop = bool(trainer.should_stop or stop)
        def on_save_checkpoint(self,trainer,module,checkpoint):
            from .native import rng_state
            checkpoint['study_rng'] = rng_state()
        def on_load_checkpoint(self,trainer,module,checkpoint):
            self.resume_rng = checkpoint.get('study_rng')
        def on_train_start(self,trainer,module):
            from .native import restore_rng
            if self.resume_rng: restore_rng(self.resume_rng)
    mc = RFDETRSegNanoConfig(num_classes=1,resolution=cfg.rfdetr_resolution)
    fields = SegmentationTrainConfig.model_fields
    if 'compute_val_loss' not in fields:
        raise RuntimeError('Installed RF-DETR lacks compute_val_loss; use the supplied environment')
    options = dict(dataset_dir=layout(data,Path(cfg.cache_root)/data['fingerprint']/'rfdetr'),
        epochs=cfg.max_epochs,batch_size=cfg.batch_size,grad_accum_steps=1,
        output_dir=str(Path(run_dir)/'framework'),
        seed=cfg.seed,lr=cfg.learning_rate,num_workers=cfg.workers,
        use_ema=False,early_stopping=False,compute_val_loss=True,
        compute_test_loss=False,multi_scale=False,expanded_scales=False,
        lr_scheduler='cosine',eval_interval=1,eval_max_dets=cfg.max_dets,
        run_test=False,clearml=False,wandb=False,mlflow=False,tensorboard=False)
    tc = SegmentationTrainConfig(**options)
    atomic_json(Path(run_dir)/'rfdetr_training_config.json',tc.model_dump(mode='json'))
    module = RFDETRModelModule(mc,tc)
    dm = RFDETRDataModule(mc,tc)
    trainer = build_trainer(tc,mc,devices=1,accelerator='gpu',
                            precision='16-mixed' if cfg.amp else '32-true',num_sanity_val_steps=0)
    # Preserve framework COCO evaluation, but own best checkpoint and stopping policy.
    from pytorch_lightning.callbacks import ModelCheckpoint
    trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c,ModelCheckpoint)
                         and type(c).__name__ not in ('BestModelCallback','RFDETREarlyStopping')]
    trainer.callbacks.extend([StudyCallback(),ModelCheckpoint(dirpath=str(folder),
        filename='epoch-{epoch}',save_last=True,save_top_k=0,every_n_epochs=1,save_on_train_epoch_end=True)])
    trainer.fit(module,dm,ckpt_path=str(last) if last.exists() else None)
    if not best.exists(): raise RuntimeError('No RF-DETR validation-selected checkpoint')
    atomic_json(Path(run_dir)/'training_complete.json',{'epochs':trainer.current_epoch,
        'best_epoch':stopper.best_epoch,'best_val_mask_AP50_95':stopper.best,
        'reason':'patience' if stopper.bad_epochs>=cfg.patience else 'max_epochs'})
    return best
