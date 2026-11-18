"""YOLO n/s/m, validation-mask-AP stopping and explicit dual-loss reporting."""
from pathlib import Path
import shutil
from .storage import atomic_json, read_json
from .tracking import EarlyStop

def train(cfg,name,adapter,data,tracker,run_dir):
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment.train import SegmentationTrainer
    folder = Path(run_dir)/'checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    last,best = folder/'last.pt',folder/'best.pt'
    stop_state = folder/'early_stop.json'
    stopper = EarlyStop(cfg.patience,cfg.min_delta)
    if last.exists():
        import torch
        saved = torch.load(last,map_location='cpu',weights_only=False)
        if 'study_state' not in saved: raise RuntimeError('Resume state missing: do not silently restart')
        stopper.load_state_dict(saved['study_state']['state'])
        atomic_json(stop_state,saved['study_state'])
        if not tracker.rows or tracker.rows[-1]['epoch'] < saved['study_row']['epoch']:
            tracker.epoch(saved['study_row'])
    class MaskTrainer(SegmentationTrainer):
        def resume_training(self,ckpt):
            super().resume_training(ckpt)
            if ckpt and self.resume and 'study_live_model' in ckpt:
                from ultralytics.utils.torch_utils import unwrap_model
                unwrap_model(self.model).load_state_dict(ckpt['study_live_model'])
                self.optimizer.load_state_dict(ckpt['study_optimizer'])
                self.study_resume_rng=ckpt['study_rng']
                self.study_resume_scheduler=ckpt['study_scheduler']
        def validate(self):
            metrics,_ = super().validate()
            # Native validator has computed validation losses and mask AP.
            return metrics,float(metrics['metrics/mAP50-95(M)'])
    model = YOLO(str(last) if last.exists() else name.replace('_seg','-seg.pt'))
    seen = set()
    def epoch_end(trainer):
        if int(trainer.batch_size)!=cfg.batch_size:
            raise RuntimeError('YOLO changed batch size after OOM. Choose a new explicit configuration/run ID.')
        epoch = trainer.epoch+1
        if epoch in seen: return
        seen.add(epoch)
        terms = {**trainer.label_loss_items(trainer.tloss,prefix='train'),**trainer.metrics}
        train_loss = sum(float(v) for k,v in terms.items() if k.startswith('train/') and 'loss' in k)
        val_keys = [k for k in terms if k.startswith('val/') and 'loss' in k]
        # Ultralytics invokes this callback once more after the final epoch while
        # stripping/exporting weights. That event is not a new training epoch.
        if not val_keys and epoch > cfg.max_epochs:
            return
        if not val_keys:
            raise RuntimeError('YOLO validator did not report validation losses')
        row = {'epoch':epoch,'train_loss':train_loss,
            'val_loss':sum(float(terms[k]) for k in val_keys),
            'val_mask_AP50_95':100*float(terms['metrics/mAP50-95(M)']),
            **{k.replace('/','_'):float(v) for k,v in terms.items()}}
        improved,stop = stopper.update(row['val_mask_AP50_95'],epoch)
        if not Path(trainer.last).exists(): raise RuntimeError('YOLO last checkpoint not saved')
        import torch
        from .native import save_checkpoint
        checkpoint=torch.load(trainer.last,map_location='cpu',weights_only=False)
        checkpoint['study_state']={'epoch':epoch,'state':stopper.state_dict()}
        checkpoint['study_row']=row
        from ultralytics.utils.torch_utils import unwrap_model
        from .native import rng_state
        checkpoint['study_live_model']={k:v.detach().cpu().clone() for k,v in unwrap_model(trainer.model).state_dict().items()}
        checkpoint['study_optimizer']=trainer.optimizer.state_dict()
        checkpoint['study_scheduler']=trainer.scheduler.state_dict()
        checkpoint['study_rng']=rng_state()
        save_checkpoint(last,checkpoint)
        if improved:
            save_checkpoint(best,checkpoint)
        atomic_json(stop_state,{'epoch':epoch,'state':stopper.state_dict()})
        tracker.epoch(row)
        trainer.stop = bool(trainer.stop or stop)
    model.add_callback('on_fit_epoch_end',epoch_end)
    def restore_start(trainer):
        if hasattr(trainer,'study_resume_rng'):
            from .native import restore_rng
            restore_rng(trainer.study_resume_rng)
            trainer.scheduler.load_state_dict(trainer.study_resume_scheduler)
    model.add_callback('on_train_start',restore_start)
    if not (last.exists() and (stopper.bad_epochs>=cfg.patience or read_json(stop_state)['epoch']>=cfg.max_epochs)):
        model.train(trainer=MaskTrainer,data=data['yolo_yaml'],epochs=cfg.max_epochs,
            batch=cfg.batch_size,imgsz=cfg.img_size,workers=cfg.workers,seed=cfg.seed,
            project=str(Path(run_dir)/'framework'),name='yolo',exist_ok=True,
            resume=str(last) if last.exists() else False,patience=0,save=True,
            val=True,plots=True,amp=cfg.amp,deterministic=False,
            optimizer='AdamW',lr0=cfg.learning_rate,cos_lr=True,
            max_det=cfg.max_dets,conf=cfg.prediction_floor)
    if not best.exists(): raise RuntimeError('No selected YOLO checkpoint')
    state = read_json(stop_state)
    atomic_json(Path(run_dir)/'training_complete.json',{'epochs':state['epoch'],
        'best_epoch':stopper.best_epoch,'best_val_mask_AP50_95':stopper.best,
        'reason':'patience' if stopper.bad_epochs>=cfg.patience else 'max_epochs'})
    return best
