"""SAMSAR prompted segmentation using the benchmark's canonical COCO data."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..evaluate import detection
from .base import ModelAdapter


def _load_reference_builder(repo_dir: str):
    repository = str(Path(repo_dir).resolve())
    if repository not in sys.path:
        sys.path.insert(0, repository)
    try:
        from samsar.modeling.build_samsar import build_samsar
    except Exception as error:
        raise RuntimeError(
            f"Cannot import SAMSAR reference implementation from {repo_dir}") from error
    return build_samsar


def _annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mask_utils
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        encoded = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    else:
        encoded = segmentation
    return mask_utils.decode(encoded).astype(np.uint8)


class CocoPromptDataset:
    """One training sample per image, containing all instance box prompts."""
    def __init__(self, annotation_file: str, image_dir: str,
                 image_size: int = 256) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        self.images = payload.get("images", [])
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            self.annotations[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        info = self.images[index]
        original_height, original_width = int(info["height"]), int(info["width"])
        with Image.open(self.image_dir / info["file_name"]) as source:
            resized = source.convert("RGB").resize((self.image_size, self.image_size),
                                                   Image.Resampling.BILINEAR)
        image = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
        masks, boxes = [], []
        for annotation in self.annotations.get(int(info["id"]), []):
            native_mask = _annotation_mask(annotation, original_height, original_width)
            scaled = np.asarray(Image.fromarray(native_mask).resize(
                (self.image_size, self.image_size), Image.Resampling.NEAREST),
                dtype=np.uint8)
            if not scaled.any():
                continue
            rows, columns = np.nonzero(scaled)
            masks.append(scaled)
            boxes.append([columns.min(), rows.min(), columns.max(), rows.max()])
        return {"image": image,
                "masks": (torch.from_numpy(np.stack(masks)).float() if masks else
                          torch.zeros((0, self.image_size, self.image_size))),
                "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "image_id": int(info["id"]),
                "original_size": (original_height, original_width)}


def collate_prompts(samples: list[dict]) -> dict:
    import torch
    return {"images": torch.stack([sample["image"] for sample in samples]),
            "masks": [sample["masks"] for sample in samples],
            "boxes": [sample["boxes"] for sample in samples],
            "image_ids": [sample["image_id"] for sample in samples],
            "original_sizes": [sample["original_size"] for sample in samples]}


def _freeze_sam_components(model) -> None:
    train_tokens = ("adapter", "cnn", "cfa", "cross_fused", "fusion")
    frozen_tokens = ("image_encoder", "prompt_encoder", "mask_decoder")
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(token in lowered for token in frozen_tokens) and not any(
                token in lowered for token in train_tokens):
            parameter.requires_grad_(False)


class SAMSAR(ModelAdapter):
    name = "samsar"
    protocol = "prompted"

    def __init__(self, cfg, dataset: str = "ssdd", repo_dir: str | None = None,
                 sam_checkpoint: str | None = None) -> None:
        super().__init__(cfg, dataset)
        self.repo_dir = repo_dir or cfg.samsar_repo
        self.sam_checkpoint = sam_checkpoint or cfg.sam_checkpoint
        self.ckpt = str(Path(self.run_dir) / "samsar_best.pt")
        self.prompt_boxes: dict[int, np.ndarray] | None = None
        self.prompt_scores: dict[int, np.ndarray] | None = None

    def _architecture_record(self) -> None:
        if not self.arch_record:
            self.record_architecture({"backbone": "SAM ViT-B",
                                      "mask_head": "frozen SAM mask decoder",
                                      "attention": "CFA",
                                      "input_size": self.cfg.samsar_img_size})

    def train(self, data: dict) -> None:
        import torch
        from torch.utils.data import DataLoader
        from .. import clearml_utils as tracking
        from .paper_modules import samsar_hybrid_loss
        build_samsar = _load_reference_builder(self.repo_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_samsar(sam_checkpoint=self.sam_checkpoint).to(device)
        _freeze_sam_components(self.model)
        self._architecture_record()
        recipe = self.training_recipe()
        dataset = CocoPromptDataset(data["coco_train"], data["train_image_dir"],
                                    self.cfg.samsar_img_size)
        loader = DataLoader(dataset, batch_size=recipe["batch_size"], shuffle=True,
                            num_workers=self.cfg.workers, collate_fn=collate_prompts)
        trainable = [parameter for parameter in self.model.parameters()
                     if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("SAMSAR exposes no trainable adapter/CNN/CFA parameters")
        optimizer = torch.optim.Adam(trainable, lr=float(recipe.get("lr", .0005)))
        torch.manual_seed(self.cfg.seed)
        best_loss = float("inf")
        for epoch in range(int(recipe["epochs"])):
            self.model.train()
            total_loss, steps = 0., 0
            for batch in loader:
                boxes = [value.to(device) for value in batch["boxes"]]
                if not any(len(value) for value in boxes):
                    continue
                targets = [value.to(device) for value in batch["masks"]]
                outputs = self.model(batch["images"].to(device), boxes)
                predicted = [item["masks"][:, 0] for item in outputs
                             if item["masks"].numel()]
                expected = [item for item in targets if item.numel()]
                if not predicted or not expected:
                    continue
                logits, truth = torch.cat(predicted), torch.cat(expected)
                count = min(len(logits), len(truth))
                objective = samsar_hybrid_loss(logits[:count], truth[:count])
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                total_loss += float(objective.detach())
                steps += 1
            average = total_loss / max(steps, 1)
            tracking.log_scalar(f"{self.dataset}/{self.name}", "train_loss",
                                average, epoch + 1)
            if average < best_loss:
                best_loss = average
                Path(self.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.ckpt)

    def load(self, data: dict) -> None:
        import torch
        build_samsar = _load_reference_builder(self.repo_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_samsar(sam_checkpoint=self.sam_checkpoint).to(device)
        _freeze_sam_components(self.model)
        if Path(self.ckpt).is_file():
            state = torch.load(self.ckpt, map_location=device)
            self.model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"Trained SAMSAR checkpoint not found: {self.ckpt}")
        self.model.eval()
        self._architecture_record()

    def set_prompts_from_detector(self, boxes_by_image: dict[int, np.ndarray],
                                  scores_by_image: dict[int, np.ndarray]) -> None:
        self.prompt_boxes = boxes_by_image
        self.prompt_scores = scores_by_image
        self.protocol = "end2end"

    @staticmethod
    def _ground_truth_boxes(payload: dict) -> dict[int, np.ndarray]:
        grouped: dict[int, list[list[float]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            x, y, width, height = map(float, annotation["bbox"])
            grouped[int(annotation["image_id"])].append(
                [x, y, x + width, y + height])
        return {image_id: np.asarray(values, dtype=np.float32)
                for image_id, values in grouped.items()}

    def predict_coco(self, data: dict) -> list[dict]:
        import torch
        from torch.nn import functional as F
        if self.model is None:
            raise RuntimeError("SAMSAR is not loaded")
        payload = json.loads(Path(data["coco_test"]).read_text(encoding="utf-8"))
        prompts = (self.prompt_boxes if self.prompt_boxes is not None
                   else self._ground_truth_boxes(payload))
        device = next(self.model.parameters()).device
        size = self.cfg.samsar_img_size
        output = []
        with torch.no_grad():
            for info in payload.get("images", []):
                boxes = prompts.get(int(info["id"]))
                if boxes is None or len(boxes) == 0:
                    continue
                with Image.open(Path(data["test_image_dir"]) / info["file_name"]) as source:
                    resized = source.convert("RGB").resize((size, size),
                                                           Image.Resampling.BILINEAR)
                image = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
                image = image.float().unsqueeze(0).to(device)
                scaled = np.asarray(boxes, dtype=np.float32).copy()
                scaled[:, [0, 2]] *= size / info["width"]
                scaled[:, [1, 3]] *= size / info["height"]
                result = self.model(image, [torch.from_numpy(scaled).to(device)])[0]
                logits = F.interpolate(result["masks"],
                                       (info["height"], info["width"]),
                                       mode="bilinear", align_corners=False)
                masks = (logits[:, 0] > 0).cpu().numpy().astype(np.uint8)
                if self.prompt_scores is None:
                    scores = result["iou_predictions"].reshape(-1).cpu().numpy()
                else:
                    scores = np.asarray(self.prompt_scores.get(int(info["id"]), []))
                for index, mask in enumerate(masks):
                    if mask.any():
                        score = float(scores[index]) if index < len(scores) else 1.
                        # IoU regression can exceed [0,1]; clip consistently on all splits.
                        score = float(np.clip(score,0.,1.))
                        if score >= self.cfg.score_threshold:
                            output.append(detection(info["id"], mask, score))
        return output

    def predict_one(self, image_path: str):
        import torch
        if self.model is None:
            raise RuntimeError("SAMSAR is not loaded")
        device = next(self.model.parameters()).device
        size = self.cfg.samsar_img_size
        with Image.open(image_path) as source:
            resized = source.convert("RGB").resize((size, size))
        image = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
        box = torch.tensor([[size * .25, size * .25, size * .75, size * .75]],
                           device=device)
        with torch.no_grad():
            return self.model(image.float().unsqueeze(0).to(device), [box])

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "img_size": self.cfg.samsar_img_size,
                "prompt_source": ("detector" if self.prompt_boxes is not None
                                  else "ground_truth_boxes"),
                "sam_checkpoint": self.sam_checkpoint}
