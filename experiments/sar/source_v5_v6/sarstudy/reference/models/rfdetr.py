"""RF-DETR segmentation adapter with an official SAHI inference variant."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from ..evaluate import detection
from .base import ModelAdapter


def _link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _materialise_roboflow_layout(data: dict, out_dir: str) -> str:
    root = Path(out_dir)
    mapping = (("train", data["coco_train"], data["train_image_dir"]),
               ("valid", data["coco_test"], data["test_image_dir"]),
               ("test", data["coco_test"], data["test_image_dir"]))
    for split, annotation_file, image_folder in mapping:
        destination = root / split
        destination.mkdir(parents=True, exist_ok=True)
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        (destination / "_annotations.coco.json").write_text(
            json.dumps(payload), encoding="utf-8")
        for image in payload.get("images", []):
            source = Path(image_folder) / image["file_name"]
            if source.is_file():
                _link_or_copy(source, destination / source.name)
    return str(root)


def _build_seg_model(variant: str = "nano", checkpoint: str | None = None):
    import rfdetr
    names = {"nano": ("RFDETRSegNano", "RFDETRSegPreview", "RFDETRSeg"),
             "small": ("RFDETRSegSmall", "RFDETRSegPreview", "RFDETRSeg"),
             "medium": ("RFDETRSegMedium", "RFDETRSegPreview", "RFDETRSeg")}
    errors = []
    for class_name in names.get(variant.lower(), ("RFDETRSegPreview", "RFDETRSeg")):
        model_class = getattr(rfdetr, class_name, None)
        if model_class is None:
            continue
        try:
            options = {}
            if checkpoint:
                # RF-DETR supports fine-tuned checkpoints through this
                # constructor argument. Some releases expose it via **kwargs,
                # which made the previous signature check reject a valid API.
                options["pretrain_weights"] = str(checkpoint)
            model = model_class(**options)
            return model, class_name
        except Exception as error:
            errors.append(f"{class_name}: {error}")
    raise RuntimeError("RF-DETR segmentation API unavailable: " + "; ".join(errors))


def _best_checkpoint(folder: Path) -> Path | None:
    preferred = list(folder.rglob("checkpoint_best_total.pth"))
    candidates = preferred or list(folder.rglob("*.pth"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


class RFDETRSeg(ModelAdapter):
    name = "rfdetr_seg"

    def __init__(self, cfg, dataset: str = "ssdd") -> None:
        super().__init__(cfg, dataset)
        self.size = cfg.rfdetr_variant
        self.implementation_class: str | None = None

    def train(self, data: dict) -> None:
        dataset_dir = _materialise_roboflow_layout(
            data, str(Path(self.cfg.work_dir) / "rfdetr_ds" / self.dataset))
        self.model, self.implementation_class = _build_seg_model(self.size)
        self.record_architecture({"variant": self.size, "backbone": "DINOv2",
                                  "detector": "RF-DETR",
                                  "mask_head": "MaskDINO-style segmentation head",
                                  "class": self.implementation_class})
        recipe = self.training_recipe()
        self.model.train(dataset_dir=dataset_dir, epochs=recipe["epochs"],
                         batch_size=recipe["batch_size"], grad_accum_steps=1,
                         resolution=self.cfg.img_size, output_dir=self.run_dir)
        checkpoint = _best_checkpoint(Path(self.run_dir))
        if checkpoint is None:
            raise FileNotFoundError(f"RF-DETR produced no checkpoint in {self.run_dir}")
        self.ckpt = str(checkpoint)

    def load(self, data: dict) -> None:
        checkpoint = self.ckpt or _best_checkpoint(Path(self.run_dir))
        if checkpoint is None:
            raise FileNotFoundError(f"No RF-DETR checkpoint in {self.run_dir}")
        self.ckpt = str(checkpoint)
        # Reconstruct from the recorded checkpoint even immediately after
        # training, so evaluation never silently uses final-epoch memory while
        # reporting the selected/best checkpoint path.
        self.model, self.implementation_class = _build_seg_model(self.size, self.ckpt)
        if not self.arch_record:
            self.record_architecture({"variant": self.size, "backbone": "DINOv2",
                                      "detector": "RF-DETR",
                                      "mask_head": "MaskDINO-style segmentation head",
                                      "class": self.implementation_class})
        optimize = getattr(self.model, "optimize_for_inference", None)
        if callable(optimize):
            optimize()

    def _predict_image(self, image: Image.Image, threshold: float | None = None):
        if self.model is None:
            raise RuntimeError("RF-DETR model is not loaded")
        return self.model.predict(image, threshold=(self.cfg.score_threshold
                                                    if threshold is None else threshold))

    def predict_coco(self, data: dict) -> list[dict]:
        payload = json.loads(Path(data["coco_test"]).read_text(encoding="utf-8"))
        output = []
        for image_info in payload.get("images", []):
            image_path = Path(data["test_image_dir"]) / image_info["file_name"]
            with Image.open(image_path) as source:
                result = self._predict_image(source.convert("RGB"))
            masks = getattr(result, "mask", None)
            if masks is None:
                continue
            confidences = np.asarray(getattr(result, "confidence", []))
            for index, raw_mask in enumerate(np.asarray(masks)):
                binary = np.asarray(raw_mask, dtype=np.uint8)
                if binary.shape != (image_info["height"], image_info["width"]):
                    import cv2
                    binary = cv2.resize(binary, (image_info["width"], image_info["height"]),
                                        interpolation=cv2.INTER_NEAREST)
                if binary.any():
                    score = float(confidences[index]) if index < len(confidences) else 1.
                    output.append(detection(image_info["id"], binary, score))
        return output

    def predict_one(self, image_path: str):
        with Image.open(image_path) as source:
            return self._predict_image(source.convert("RGB"))

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "rfdetr_variant": self.size,
                "rfdetr_class": self.implementation_class}


class RFDETRSegSAHI(RFDETRSeg):
    name = "rfdetr_seg_sahi"
    reuses = "rfdetr_seg"

    def train(self, data: dict) -> None:
        raise RuntimeError("SAHI variants reuse their base model checkpoint")

    def load(self, data: dict) -> None:
        # The runner transfers the already-loaded base model. If invoked alone,
        # restore the base checkpoint explicitly.
        if self.model is None:
            base_folder = Path(self.cfg.runs_dir(self.dataset)) / "rfdetr_seg"
            checkpoint = _best_checkpoint(base_folder)
            if checkpoint is None:
                raise FileNotFoundError(f"No base RF-DETR checkpoint in {base_folder}")
            self.ckpt = str(checkpoint)
            self.model, self.implementation_class = _build_seg_model(self.size, self.ckpt)
        from .sahi_wrap import build_sahi_rfdetr
        self.model = build_sahi_rfdetr(self.model, self.ckpt, self.cfg)
        self.record_architecture({"variant": self.size, "backbone": "DINOv2",
                                  "detector": "RF-DETR",
                                  "mask_head": "MaskDINO-style segmentation head",
                                  "inference": "SAHI"})

    def predict_coco(self, data: dict) -> list[dict]:
        from .sahi_wrap import sliced_predict_coco
        return sliced_predict_coco(self.model, data, self.cfg, "rfdetr+sahi")

    def predict_one(self, image_path: str):
        from .sahi_wrap import sliced_predict_one
        return sliced_predict_one(self.model, image_path, self.cfg)

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "reuses_checkpoint_of": self.reuses,
                "sahi_slice": self.cfg.sahi_slice,
                "sahi_overlap": self.cfg.sahi_overlap}

