"""Prediction wrappers; training is exclusively controlled by the new runners."""
from pathlib import Path
import json
from .reference.config import Config

def bridge(cfg, path):
    return Config(work_dir=str(path), img_size=cfg.img_size,
        samsar_img_size=cfg.samsar_img_size, batch_size=cfg.batch_size,
        epochs=cfg.max_epochs, workers=cfg.workers, seed=cfg.seed,
        score_threshold=cfg.prediction_floor, max_dets=cfg.max_dets,
        samsar_repo=cfg.samsar_repo, sam_checkpoint=cfg.sam_checkpoint,
        use_paper_training_config=False, use_clearml=False)

def build_adapter(cfg, dataset, name, path):
    if name.startswith('yolo26'):
        from .reference.models.yolo26 import YOLO26Seg
        cls = YOLO26Seg
    elif name == 'rfdetr_seg':
        from .reference.models.rfdetr import RFDETRSeg
        cls = RFDETRSeg
    elif name == 'samsar':
        from .reference.models.samsar import SAMSAR
        cls = SAMSAR
    elif name == 'ship_contour':
        from .reference.models.ship_contour import ShipContour
        cls = ShipContour
    elif name == 'swininsseg':
        from .reference.models.swininsseg import SwinInsSeg
        cls = SwinInsSeg
    elif name == 'solov2_gala':
        from .reference.models.solov2_gala import SOLOv2GALA
        cls = SOLOv2GALA
    else:
        raise ValueError(name)
    def no_legacy_training(self, data):
        raise RuntimeError('Use sarstudy.runner.run_experiment; legacy training is disabled')
    adapted = type('StudyAdapter', (cls,), {'run_dir':property(lambda self:str(path)),
                                          'train':no_legacy_training})
    adapter = adapted(bridge(cfg,path), dataset)
    adapter.study_cfg = cfg
    if name.startswith('yolo26'):
        adapter.pretrained = name.replace('_seg','-seg.pt')
    return adapter

def predict(adapter, data):
    """No resumable cross-split prediction cache: always use the current model."""
    if adapter.name != 'ship_contour':
        return adapter.predict_coco(data)
    from .reference.evaluate import detection
    payload = json.loads(Path(data['coco_test']).read_text())
    output = []
    for info in payload['images']:
        masks, scores = adapter._predict_native(str(Path(data['test_image_dir'])/info['file_name']))
        output.extend(detection(info['id'],mask,float(score)) for mask,score in zip(masks,scores) if mask.any())
    return output

def load_selected(adapter, name, path):
    import torch
    if name.startswith('yolo26'):
        from ultralytics import YOLO
        adapter.model = YOLO(str(path))
    elif name == 'rfdetr_seg':
        from rfdetr import RFDETRSegNano
        adapter.model = RFDETRSegNano(pretrain_weights=str(path), num_classes=1,
                                      resolution=adapter.study_cfg.rfdetr_resolution)
    else:
        state = torch.load(path, map_location='cpu', weights_only=False)
        adapter.model.load_state_dict(state['model'], strict=True)
        adapter.model.eval()
    adapter.ckpt = str(path)
