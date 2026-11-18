"""Paper-spec ResNet-50 SOLOv2 with GALA-enhanced FPN outputs."""
from __future__ import annotations

import os
from pathlib import Path

from .base import ModelAdapter
from .mmdet_utils import (coco_predictions, infer_one, newest_checkpoint,
                          mmdet_base_config, require_training_api,
                          restore_runner)
from .paper_modules import EFPN, GALA


def build_gala(channels: int, reduction: int = 16):
    return GALA(channels, reduction)


class GALANeckWrapper:
    @staticmethod
    def patch(neck, channels: int = 256, num_levels: int = 5):
        if hasattr(neck, "gala_efpn"):
            return neck
        neck.add_module("gala_efpn", EFPN(channels, num_levels))
        original_forward = neck.forward
        def enhanced_forward(inputs):
            features = original_forward(inputs)
            neck.gala_efpn.to(features[0].device)
            return neck.gala_efpn(features)
        neck.forward = enhanced_forward
        return neck


def attach_gala(model):
    GALANeckWrapper.patch(model.neck, channels=256, num_levels=5)
    return model


class SOLOv2GALA(ModelAdapter):
    name = "solov2_gala"

    def __init__(self, cfg, dataset: str = "ssdd",
                 use_gala: bool = True) -> None:
        super().__init__(cfg, dataset)
        if not use_gala:
            raise ValueError("solov2_gala cannot be labelled as such with GALA disabled")
        self.config_path = str(Path(self.run_dir) / f"solov2_gala_{dataset}.py")

    def _write_config(self, data: dict) -> str:
        recipe = self.training_recipe()
        optimizer = recipe.get("optimizer", "SGD")
        learning_rate = recipe.get("lr", .01)
        base_config = mmdet_base_config("solov2/solov2_r50_fpn_1x_coco.py")
        text = f'''_base_ = r"{base_config}"
metainfo = dict(classes=("ship",))
model = dict(mask_head=dict(num_classes=1))
train_dataloader = dict(batch_size={recipe['batch_size']}, num_workers={self.cfg.workers},
    dataset=dict(metainfo=metainfo, ann_file=r"{data['coco_train']}",
                 data_prefix=dict(img=r"{data['train_image_dir']}/")))
val_dataloader = dict(batch_size=1, num_workers={self.cfg.workers},
    dataset=dict(metainfo=metainfo, ann_file=r"{data['coco_test']}",
                 data_prefix=dict(img=r"{data['test_image_dir']}/")))
test_dataloader = val_dataloader
val_evaluator = dict(type="CocoMetric", ann_file=r"{data['coco_test']}", metric=["segm"])
test_evaluator = val_evaluator
optim_wrapper = dict(optimizer=dict(type="{optimizer}", lr={learning_rate},
                                    momentum=0.9, weight_decay=0.0001))
train_cfg = dict(max_epochs={recipe['epochs']}, val_interval=5)
randomness = dict(seed={self.cfg.seed}, deterministic=False)
work_dir = r"{self.run_dir}"
'''
        Path(self.config_path).write_text(text, encoding="utf-8")
        return self.config_path

    def _record(self) -> None:
        if not self.arch_record:
            self.record_architecture({"backbone": "ResNet-50",
                                      "detector": "SOLOv2",
                                      "mask_head": "SOLOv2 dynamic mask head",
                                      "attention": "GALA EFPN"})

    def train(self, data: dict) -> None:
        MMConfig, Runner = require_training_api()
        config = MMConfig.fromfile(self._write_config(data))
        runner = Runner.from_cfg(config)
        attach_gala(runner.model)
        self._record()
        runner.train()
        self.model = runner.model
        self.model.cfg = config
        self.ckpt = newest_checkpoint(self.run_dir)

    def load(self, data: dict) -> None:
        if not Path(self.config_path).is_file():
            self._write_config(data)
        self.ckpt = self.ckpt or newest_checkpoint(self.run_dir)
        self.model = restore_runner(self.config_path, attach_gala, self.ckpt)
        self._record()

    def predict_coco(self, data: dict) -> list[dict]:
        return coco_predictions(self.model, data, self.cfg.score_threshold)

    def predict_one(self, image_path: str):
        return infer_one(self.model, image_path)

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "gala_enabled": True,
                "backbone": "ResNet-50+FPN", "head": "SOLOv2"}

