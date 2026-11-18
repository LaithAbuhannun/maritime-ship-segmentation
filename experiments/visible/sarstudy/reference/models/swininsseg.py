"""Paper-spec Swin-T → MKA → FPN → SOLOv2 implementation."""
from __future__ import annotations

import os
from pathlib import Path

from .base import ModelAdapter
from .mmdet_utils import (coco_predictions, infer_one, newest_checkpoint,
                          mmdet_base_config, require_training_api,
                          restore_runner)

SWIN_ARCH = {
    "tiny": {"embed_dims": 96, "depths": (2, 2, 6, 2),
             "num_heads": (3, 6, 12, 24)},
    "small": {"embed_dims": 96, "depths": (2, 2, 18, 2),
              "num_heads": (3, 6, 12, 24)},
}


def attach_mka(model, in_channels=(96, 192, 384, 768),
               out_channels: int = 256):
    """Register four MKA blocks directly after the four Swin stage outputs."""
    import torch.nn as nn
    from .paper_modules import MKA
    backbone = model.backbone
    if hasattr(backbone, "mka_blocks"):
        return model
    backbone.add_module("mka_blocks", nn.ModuleList(
        MKA(channels, out_channels) for channels in in_channels))
    original_forward = backbone.forward

    def enhanced_forward(inputs):
        features = original_forward(inputs)
        if len(features) != len(backbone.mka_blocks):
            raise RuntimeError("Swin backbone did not return four feature levels")
        return tuple(block.to(feature.device)(feature) for block, feature in
                     zip(backbone.mka_blocks, features))
    backbone.forward = enhanced_forward
    return model


class SwinInsSeg(ModelAdapter):
    name = "swininsseg"

    def __init__(self, cfg, dataset: str = "ssdd") -> None:
        super().__init__(cfg, dataset)
        self.variant = cfg.swin_variant
        self.config_path = str(Path(self.run_dir) / f"swininsseg_{dataset}.py")

    def _write_config(self, data: dict) -> str:
        architecture = SWIN_ARCH.get(self.variant)
        if architecture is None:
            raise ValueError(f"Unsupported Swin variant: {self.variant}")
        recipe = self.training_recipe()
        base_config = mmdet_base_config("solov2/solov2_r50_fpn_1x_coco.py")
        text = f'''_base_ = r"{base_config}"
metainfo = dict(classes=("ship",))
model = dict(
    backbone=dict(_delete_=True, type="SwinTransformer",
        embed_dims={architecture['embed_dims']}, depths={architecture['depths']},
        num_heads={architecture['num_heads']}, window_size=7, mlp_ratio=4,
        qkv_bias=True, drop_path_rate=0.2, patch_norm=True,
        out_indices=(0, 1, 2, 3), with_cp=False,
        init_cfg=dict(type="Pretrained",
            checkpoint="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"),
        convert_weights=True),
    neck=dict(in_channels=[256, 256, 256, 256]),
    mask_head=dict(num_classes=1,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                      alpha=0.25, loss_weight=1.0),
        loss_mask=dict(type="DiceLoss", use_sigmoid=True, loss_weight=3.0)))
train_dataloader = dict(batch_size={recipe['batch_size']}, num_workers={self.cfg.workers},
    dataset=dict(metainfo=metainfo, ann_file=r"{data['coco_train']}",
                 data_prefix=dict(img=r"{data['train_image_dir']}/")))
val_dataloader = dict(batch_size=1, num_workers={self.cfg.workers},
    dataset=dict(metainfo=metainfo, ann_file=r"{data['coco_test']}",
                 data_prefix=dict(img=r"{data['test_image_dir']}/")))
test_dataloader = val_dataloader
val_evaluator = dict(type="CocoMetric", ann_file=r"{data['coco_test']}", metric=["segm"])
test_evaluator = val_evaluator
optim_wrapper = dict(optimizer=dict(type="{recipe.get('optimizer', 'SGD')}",
    lr={recipe.get('lr', .001)}, momentum={recipe.get('momentum', .9)},
    weight_decay={recipe.get('weight_decay', .0001)}))
param_scheduler = [dict(type="MultiStepLR", by_epoch=True,
    milestones={recipe.get('lr_milestones', [8, 11])},
    gamma={recipe.get('lr_gamma', .1)})]
train_cfg = dict(max_epochs={recipe['epochs']}, val_interval=5)
randomness = dict(seed={self.cfg.seed}, deterministic=False)
work_dir = r"{self.run_dir}"
'''
        Path(self.config_path).write_text(text, encoding="utf-8")
        return self.config_path

    def _record(self) -> None:
        if not self.arch_record:
            self.record_architecture({"backbone": "Swin-T", "variant": self.variant,
                                      "detector": "SOLOv2",
                                      "mask_head": "SOLOv2 dynamic mask head",
                                      "attention": "MKA"})

    def train(self, data: dict) -> None:
        MMConfig, Runner = require_training_api()
        config = MMConfig.fromfile(self._write_config(data))
        runner = Runner.from_cfg(config)
        attach_mka(runner.model)
        self._record()
        runner.train()
        self.model = runner.model
        self.model.cfg = config
        self.ckpt = newest_checkpoint(self.run_dir)

    def load(self, data: dict) -> None:
        if not Path(self.config_path).is_file():
            self._write_config(data)
        self.ckpt = self.ckpt or newest_checkpoint(self.run_dir)
        self.model = restore_runner(self.config_path, attach_mka, self.ckpt)
        self._record()

    def predict_coco(self, data: dict) -> list[dict]:
        return coco_predictions(self.model, data, self.cfg.score_threshold)

    def predict_one(self, image_path: str):
        return infer_one(self.model, image_path)

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "swin_variant": self.variant,
                "head": "SOLOv2", "attention": "MKA(1,3,5,7)+CBAM",
                "loss": "Focal(alpha=.25,gamma=2)+3*Dice"}

