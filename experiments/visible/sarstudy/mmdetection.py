"""Use MMDetection architectures with explicit validation-loss data loading."""
from pathlib import Path
from copy import deepcopy
from .storage import atomic_json

def build_model_loaders(cfg,name,adapter,data):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from .reference.models.swininsseg import attach_mka
    from .reference.models.solov2_gala import attach_gala
    register_all_modules(init_default_scope=True)
    # The legacy generator sees validation, never test, as its evaluation input.
    alias = {**data,'coco_test':data['coco_val'],'test_image_dir':data['val_image_dir']}
    config = Config.fromfile(adapter._write_config(alias))
    config.model.test_cfg.score_thr = cfg.prediction_floor
    config.model.test_cfg.max_per_img = cfg.max_dets
    loss_pipeline = [dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations',with_bbox=True,with_mask=True),
        dict(type='Resize',scale=(cfg.img_size,cfg.img_size),keep_ratio=True),
        dict(type='PackDetInputs')]
    inference_pipeline = [dict(type='LoadImageFromFile'),
        dict(type='Resize',scale=(cfg.img_size,cfg.img_size),keep_ratio=True),
        dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    def loader(split, training):
        return dict(batch_size=cfg.batch_size if training else 1,
            num_workers=cfg.workers,persistent_workers=False,
            sampler=dict(type='DefaultSampler',shuffle=training),
            dataset=dict(type='CocoDataset',metainfo=dict(classes=('ship',)),
                ann_file=data[f'coco_{split}'],data_prefix=dict(img=data[f'{split}_image_dir']+'/'),
                filter_cfg=None,test_mode=not training,pipeline=deepcopy(loss_pipeline)))
    config.train_dataloader = loader('train',True)
    config.val_dataloader = loader('val',False)
    config.val_evaluator = dict(type='CocoMetric',ann_file=data['coco_val'],metric='segm')
    # This pipeline is used only for inference_detector, without reading test labels.
    config.test_dataloader = deepcopy(config.val_dataloader)
    config.test_dataloader.dataset.pipeline = inference_pipeline
    config.test_evaluator = deepcopy(config.val_evaluator)
    config.test_pipeline = inference_pipeline
    config.dump(adapter.config_path)
    model = MODELS.build(config.model)
    model.init_weights()
    (attach_mka if name=='swininsseg' else attach_gala)(model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.cfg = config
    model.dataset_meta = {'classes':('ship',)}
    adapter.model = model
    atomic_json(Path(adapter.run_dir)/'architecture.json',{'name':name,
        'implementation':'baseline paper-guided reimplementation; not independently revalidated',
        'optimizer':'SGD' if name=='solov2_gala' else 'AdamW','learning_rate':cfg.learning_rate,'schedule':'cosine',
        'training_augmentation':'resize only','loss_validation_mode':'eval model; mask-padding preprocessor'})
    return (Runner.build_dataloader(config.train_dataloader,seed=cfg.seed),
            Runner.build_dataloader(config.val_dataloader,seed=cfg.seed))
