"""One model/dataset per process; immutable identity and split-specific outputs."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from .config import StudyConfig
from .storage import atomic_json, read_json, file_hash, require_identity, fingerprint
from .data import prepare, prepare_generalization, evaluation_view
from .adapters import build_adapter, load_selected, predict
from .metrics import evaluate_masks, evaluate_boxes, choose_threshold, capped_predictions
from .tracking import Tracker

# Colab exports a notebook-only backend that is unavailable inside the isolated
# subprocess environment. Training and saved figures use the headless backend.
os.environ['MPLBACKEND'] = 'Agg'

def preflight(name):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU is required for training')
    gpu = torch.cuda.get_device_name(0)
    if name in ('swininsseg','solov2_gala'):
        if torch.cuda.get_device_capability(0)[0] >= 10:
            raise RuntimeError('This pinned MMDetection environment is for A100/T4, not Blackwell')
        import mmcv.ops
        boxes = torch.tensor([[0.,0.,10.,10.]],device='cuda')
        mmcv.ops.nms(boxes,torch.ones(1,device='cuda'),.5)
    # Verify CUDA kernels, not just device detection.
    x = torch.ones((16,16),device='cuda'); (x@x).sum().item(); torch.cuda.synchronize()
    versions = {}
    for package in ('torch','torchvision','numpy','ultralytics','rfdetr','mmengine','mmcv','mmdet','clearml'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    return {'python':sys.version,'platform':platform.platform(),'gpu':gpu,
            'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'cuda':torch.version.cuda,'versions':versions}

def code_fingerprint():
    root = Path(__file__).parent
    return fingerprint({str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob('*.py'))})

def smoke_signature(cfg):
    return fingerprint({k:v for k,v in cfg.as_dict().items() if k not in
        ('experiment','smoke','max_epochs','patience','models','datasets','cache_root')})

def check_predictions(data, predictions):
    payload = read_json(data['coco_test'])
    images = {int(x['id']):x for x in payload['images']}
    for p in predictions:
        if int(p['image_id']) not in images:
            raise ValueError('Prediction image ID does not belong to the evaluation split')
        if not math.isfinite(float(p['score'])): raise ValueError('Non-finite prediction score')
        info = images[int(p['image_id'])]
        if list(p['segmentation']['size']) != [info['height'],info['width']]:
            raise ValueError('Predicted mask dimensions differ from the original image')
    return predictions

def get_predictions(adapter, data, destination, checkpoint_hash, max_dets):
    identity = {'split_fingerprint':data['fingerprint'],
                'split':data['evaluation_split'],'checkpoint_sha256':checkpoint_hash}
    destination = Path(destination)
    # These files are exclusively owned by this run and this split.
    meta_path = destination.with_suffix('.identity.json')
    if destination.exists() and meta_path.exists() and read_json(meta_path) == identity:
        return check_predictions(data,read_json(destination))
    print(f"Predicting {data['dataset']}/{data['evaluation_split']}",flush=True)
    result = capped_predictions(check_predictions(data,predict(adapter,data)),max_dets)
    atomic_json(destination,result); atomic_json(meta_path,identity)
    return result

def fps(adapter,data,cfg):
    """Batch-1 end-to-end: local image read through original-size mask/RLE output.

    Includes per-image manifest parsing, excludes manifest creation and metrics.
    SAMSAR receives the same ground-truth boxes as its prompted evaluation.
    """
    import torch
    payload = read_json(data['coco_test'])
    images = payload['images'][:min(cfg.fps_images,4) if cfg.smoke else cfg.fps_images]
    if not images: return None
    views = []
    with tempfile.TemporaryDirectory(prefix='sar-fps-') as tmp:
        for i,info in enumerate(images):
            path = Path(tmp)/f'{i}.json'
            atomic_json(path,{**payload,'images':[info],
                'annotations':[a for a in payload['annotations'] if a['image_id']==info['id']]})
            views.append({**data,'coco_test':str(path)})
        for i in range(min(cfg.fps_warmup,2) if cfg.smoke else cfg.fps_warmup):
            predict(adapter,views[i%len(views)])
        torch.cuda.synchronize(); start=time.perf_counter()
        for view in views: predict(adapter,view)
        torch.cuda.synchronize(); seconds=time.perf_counter()-start
    return len(views)/seconds

def parameters(adapter):
    model = adapter.model
    for _ in range(3):
        if hasattr(model,'parameters'): return sum(p.numel() for p in model.parameters())/1e6
        model = getattr(model,'model',None)
    return None

def run_experiment(cfg,dataset,name):
    environment = preflight(name)
    folder = cfg.run_dir(dataset,name); folder.mkdir(parents=True,exist_ok=True)
    # Keep environment and source code in the resume identity: no silent changed-code resume.
    configuration = {k:v for k,v in cfg.as_dict().items() if k not in
                     ('models','datasets','clearml','clearml_project','cache_root')}
    if name=='samsar':
        source=Path(cfg.samsar_repo)
        if not (source/'samsar/modeling/build_samsar.py').is_file():
            raise FileNotFoundError('SAMSAR source is missing; run the prerequisite cell')
        configuration['samsar_source_sha256']=fingerprint({str(p.relative_to(source)):file_hash(p)
                                              for p in sorted(source.rglob('*.py'))})
        configuration['sam_initialization_sha256']=file_hash(cfg.sam_checkpoint)
    require_identity(folder/'identity.json',{'config':configuration,'dataset':dataset,
        'model':name,'code_sha256':code_fingerprint(),'versions':environment['versions']})
    atomic_json(folder/'config.json',cfg.as_dict())
    atomic_json(folder/'environment.json',environment)
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (folder/'requirements-frozen.txt').write_text(freeze)
    tracker = Tracker(cfg,dataset,name,folder)
    try:
        data = prepare(cfg,dataset)
        atomic_json(folder/'data.json',data)
        tracker.artifact('split_manifest',data['manifest_path'])
        adapter = build_adapter(cfg,dataset,name,folder)
        suffix = 'pth' if name=='rfdetr_seg' else 'pt'
        selected = folder/'checkpoints'/f'best.{suffix}'
        if not (folder/'training_complete.json').exists():
            if name.startswith('yolo26'):
                from .yolo import train
            elif name=='rfdetr_seg':
                from .rfdetr import train
            else:
                from .native import train
            selected = train(cfg,name,adapter,data,tracker,folder)
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()
        elif name not in ('rfdetr_seg',) and not name.startswith('yolo26'):
            from .native import setup
            setup(cfg,name,adapter,data)
        if not selected.exists(): raise FileNotFoundError(selected)
        load_selected(adapter,name,selected)
        checkpoint_hash = file_hash(selected)
        provenance = {'model':name,'training_dataset':dataset,'experiment':cfg.experiment,
            'clearml_task_id':tracker.task.id if tracker.task else None,
            'split_fingerprint':data['fingerprint'],'checkpoint_sha256':checkpoint_hash,
            'protocol':'ground_truth_box_prompted' if name=='samsar' else 'end2end',
            'params_M':parameters(adapter),'gpu':environment['gpu'],
            'fps_protocol':'batch1_local_image_to_original_mask_rle_framework_default_precision',
            'threshold_selected_on':'validation',
            'training':read_json(folder/'training_complete.json'),
            'smoke':cfg.smoke}
        val = get_predictions(adapter,evaluation_view(data,'val'),folder/'predictions/val.json',
                              checkpoint_hash,cfg.max_dets)
        threshold = choose_threshold(data['coco_val'],val)
        atomic_json(folder/'threshold.json',{'selected_on':'val','value':threshold,
                                            'split_fingerprint':data['fingerprint']})
        tracker.metrics('val',{**provenance,**evaluate_masks(data['coco_val'],val,threshold,cfg.max_dets)})
        test = get_predictions(adapter,evaluation_view(data,'test'),folder/'predictions/test.json',
                               checkpoint_hash,cfg.max_dets)
        print('Evaluating test split', flush=True)
        values = {**provenance,**evaluate_masks(data['coco_test'],test,threshold,cfg.max_dets),
                  'dataset':dataset}
        # Smoke verifies training, validation, test inference, metrics, persistence
        # and ClearML logging. Timing and cross-dataset work belong to final runs.
        if not cfg.smoke:
            print('Benchmarking test FPS', flush=True)
            values['FPS'] = fps(adapter,evaluation_view(data,'test'),cfg)
        tracker.metrics('test',values)
        if not cfg.smoke and dataset=='ssdd' and name!='samsar':
            print('Evaluating LS-SSD generalization', flush=True)
            general = prepare_generalization(cfg)
            predictions = get_predictions(adapter,general,folder/'predictions/ls_ssdd.json',
                                           checkpoint_hash,cfg.max_dets)
            tracker.metrics('ls_ssdd',{**provenance,'dataset':'ls_ssdd',
                'split_fingerprint':general['fingerprint'],'selection':'ship-containing images only',
                **evaluate_boxes(general['gt_boxes'],predictions,threshold,cfg.max_dets),
                'FPS':fps(adapter,general,cfg)})
        from .reports import plot_history
        for path in plot_history(folder): tracker.artifact(path.name,path)
        small_artifacts = [folder/'environment.json',folder/'requirements-frozen.txt',
                           folder/'threshold.json',folder/'training_complete.json']
        for file in small_artifacts:
            tracker.artifact(file.name,file)
        if not cfg.smoke:
            tracker.artifact(selected.name, selected)
            for file in (folder/'checkpoints').glob('last.*'): tracker.artifact(file.name,file)
            for file in (folder/'predictions').glob('*.json'): tracker.artifact('prediction_'+file.name,file)
        tracker.close()
        atomic_json(folder/'evaluation_complete.json',{'checkpoint_sha256':checkpoint_hash,'smoke':cfg.smoke,
            'code_sha256':code_fingerprint(),'smoke_signature':smoke_signature(cfg)})
        print('COMPLETE:',folder,flush=True)
    except BaseException:
        (folder/'failure.txt').write_text(traceback.format_exc())
        if tracker.task:
            try:
                tracker.artifact('failure',folder/'failure.txt')
                tracker.task.mark_failed(status_reason='See failure.txt and console')
                tracker.close()
            except Exception: pass
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--dataset',required=True,choices=['ssdd','hrsid'])
    parser.add_argument('--model',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=StudyConfig(**read_json(args.config))
    if args.smoke:
        from dataclasses import replace
        cfg=replace(cfg,experiment=cfg.experiment+'_smoke',smoke=True,max_epochs=2,patience=2)
    elif not cfg.smoke:
        marker=cfg.run_dir(args.dataset,args.model).parent/(cfg.experiment+'_smoke')/'evaluation_complete.json'
        if not marker.exists() or read_json(marker).get('code_sha256')!=code_fingerprint() or read_json(marker).get('smoke_signature')!=smoke_signature(cfg):
            raise RuntimeError('Complete a smoke run using this code and configuration before long training')
    run_experiment(cfg,args.dataset,args.model)

if __name__=='__main__': main()
