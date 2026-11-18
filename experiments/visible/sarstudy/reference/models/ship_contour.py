"""DLA/ECA/Mish CenterNet with circular-convolution contour refinement."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..evaluate import detection
from .base import ModelAdapter

N_VERTICES = 128


def box_to_octagon(box: np.ndarray) -> np.ndarray:
    left, top, right, bottom = map(float, box)
    width, height = right - left, bottom - top
    return np.asarray([
        [left + .25 * width, top], [left + .75 * width, top],
        [right, top + .25 * height], [right, top + .75 * height],
        [left + .75 * width, bottom], [left + .25 * width, bottom],
        [left, top + .75 * height], [left, top + .25 * height]], dtype=np.float32)


def resample_contour(points: np.ndarray, n: int = N_VERTICES) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 2:
        raise ValueError("A contour needs at least two points")
    closed = np.concatenate((polygon, polygon[:1]), axis=0)
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 0:
        return np.repeat(polygon[:1], n, axis=0)
    cumulative = np.concatenate(([0.], np.cumsum(lengths)))
    targets = np.linspace(0, perimeter, n, endpoint=False)
    output = np.empty((n, 2), dtype=np.float32)
    segment = 0
    for index, distance in enumerate(targets):
        while segment + 1 < len(cumulative) and cumulative[segment + 1] <= distance:
            segment += 1
        fraction = ((distance - cumulative[segment]) / max(lengths[segment], 1e-8))
        output[index] = closed[segment] + fraction * (closed[segment + 1] - closed[segment])
    return output


def gaussian_radius(height: float, width: float, min_overlap: float = .7) -> float:
    a1, b1 = 1., height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    radius1 = (b1 + math.sqrt(max(0., b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2 = 4., 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    radius2 = (b2 + math.sqrt(max(0., b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    radius3 = (b3 + math.sqrt(max(0., b3 * b3 - 4 * a3 * c3))) / (2 * a3)
    return max(0., min(radius1, radius2, radius3))


def draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: float) -> None:
    radius_i = max(0, int(radius))
    diameter = 2 * radius_i + 1
    coordinates = np.arange(diameter, dtype=np.float32) - radius_i
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    sigma = max(diameter / 6, 1e-6)
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    height, width = heatmap.shape
    left, right = min(cx, radius_i), min(width - cx - 1, radius_i)
    top, bottom = min(cy, radius_i), min(height - cy - 1, radius_i)
    if min(left, right, top, bottom) < 0:
        return
    target = heatmap[cy - top:cy + bottom + 1, cx - left:cx + right + 1]
    source = gaussian[radius_i - top:radius_i + bottom + 1,
                      radius_i - left:radius_i + right + 1]
    np.maximum(target, source, out=target)


def _build_network():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from .paper_modules import ECA, Mish

    class BasicBlock(nn.Module):
        def __init__(self, input_channels, output_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride, 1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.conv2 = nn.Conv2d(output_channels, output_channels, 3, 1, 1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(output_channels)
            self.activation = Mish()
            self.eca = ECA(output_channels)
            self.skip = (nn.Identity() if stride == 1 and input_channels == output_channels
                         else nn.Sequential(nn.Conv2d(input_channels, output_channels, 1,
                                                      stride, bias=False),
                                            nn.BatchNorm2d(output_channels)))

        def forward(self, value):
            residual = self.skip(value)
            value = self.activation(self.bn1(self.conv1(value)))
            value = self.eca(self.bn2(self.conv2(value)))
            return self.activation(value + residual)

    class Root(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, output_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(output_channels)
            self.activation = Mish()

        def forward(self, *children):
            return self.activation(self.bn(self.conv(torch.cat(children, dim=1))))

    class Tree(nn.Module):
        def __init__(self, levels, input_channels, output_channels, stride=1,
                     root_channels=0, level_root=False):
            super().__init__()
            self.levels, self.level_root = levels, level_root
            self.downsample = nn.MaxPool2d(stride, stride) if stride > 1 else nn.Identity()
            self.project = (nn.Sequential(nn.Conv2d(input_channels, output_channels, 1,
                                                    bias=False),
                                          nn.BatchNorm2d(output_channels))
                            if input_channels != output_channels else nn.Identity())
            if levels == 1:
                self.tree1 = BasicBlock(input_channels, output_channels, stride)
                self.tree2 = BasicBlock(output_channels, output_channels)
                root_input = 2 * output_channels + root_channels
                if level_root:
                    root_input += input_channels
                self.root = Root(root_input, output_channels)
            else:
                self.tree1 = Tree(levels - 1, input_channels, output_channels, stride)
                child_channels = root_channels + output_channels
                if level_root:
                    child_channels += input_channels
                self.tree2 = Tree(levels - 1, output_channels, output_channels,
                                  root_channels=child_channels, level_root=True)

        def forward(self, value, children=None):
            children = [] if children is None else list(children)
            bottom = self.downsample(value)
            residual = self.project(bottom)
            if self.level_root:
                children.append(bottom)
            first = self.tree1(value)
            if self.levels == 1:
                second = self.tree2(first)
                return self.root(second, first, *children)
            children.append(first)
            return self.tree2(first, children=children)

    class DLA60(nn.Module):
        channels = (16, 32, 64, 128, 256, 512)
        levels = (1, 1, 1, 2, 3, 1)

        def __init__(self):
            super().__init__()
            self.base = nn.Sequential(nn.Conv2d(3, 16, 7, 1, 3, bias=False),
                                      nn.BatchNorm2d(16), Mish())
            self.level0 = self._conv_level(16, 16, self.levels[0])
            self.level1 = self._conv_level(16, 32, self.levels[1], stride=2)
            self.level2 = Tree(self.levels[2], 32, 64, stride=2)
            self.level3 = Tree(self.levels[3], 64, 128, stride=2, level_root=True)
            self.level4 = Tree(self.levels[4], 128, 256, stride=2, level_root=True)
            self.level5 = Tree(self.levels[5], 256, 512, stride=2, level_root=True)
            self.laterals = nn.ModuleList(nn.Conv2d(channels, 64, 1)
                                          for channels in self.channels[2:])
            self.fuse = nn.Sequential(nn.Conv2d(64 * 4, 128, 3, padding=1),
                                      nn.BatchNorm2d(128), Mish())

        @staticmethod
        def _conv_level(input_channels, output_channels, count, stride=1):
            layers = []
            for index in range(count):
                layers.append(BasicBlock(input_channels if index == 0 else output_channels,
                                         output_channels, stride if index == 0 else 1))
            return nn.Sequential(*layers)

        def forward(self, image):
            value = self.base(image)
            value = self.level0(value)
            value = self.level1(value)
            features = []
            for level in (self.level2, self.level3, self.level4, self.level5):
                value = level(value)
                features.append(value)
            target_size = features[0].shape[-2:]
            projected = [lateral(feature) for lateral, feature in
                         zip(self.laterals, features)]
            projected = [projected[0], *[F.interpolate(item, target_size,
                                                       mode="bilinear",
                                                       align_corners=False)
                                        for item in projected[1:]]]
            return self.fuse(torch.cat(projected, dim=1))

    class SnakeBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, 9, padding=4,
                                  padding_mode="circular")
            self.norm = nn.BatchNorm1d(channels)
            self.activation = Mish()

        def forward(self, value):
            return self.activation(self.norm(self.conv(value)))

    class ContourEvolution(nn.Module):
        def __init__(self, feature_channels=128, hidden=128, iterations=3):
            super().__init__()
            self.iterations = iterations
            self.input = nn.Conv1d(feature_channels + 2, hidden, 1)
            self.blocks = nn.Sequential(*[SnakeBlock(hidden) for _ in range(4)])
            self.offset = nn.Conv1d(hidden, 2, 1)

        def forward(self, feature_map, contours):
            if contours.numel() == 0:
                return contours
            batch, instances, vertices, _ = contours.shape
            result = contours
            height, width = feature_map.shape[-2:]
            for _ in range(self.iterations):
                normalised = result.clone()
                normalised[..., 0] = normalised[..., 0] / max(width - 1, 1) * 2 - 1
                normalised[..., 1] = normalised[..., 1] / max(height - 1, 1) * 2 - 1
                grid = normalised.reshape(batch, instances * vertices, 1, 2)
                sampled = F.grid_sample(feature_map, grid, align_corners=True)
                sampled = sampled.squeeze(-1).reshape(batch, -1, instances, vertices)
                coordinates = normalised.permute(0, 3, 1, 2)
                coordinates = coordinates.reshape(batch, 2, instances, vertices)
                combined = torch.cat((sampled, coordinates), dim=1)
                combined = combined.permute(0, 2, 1, 3).reshape(
                    batch * instances, -1, vertices)
                offsets = self.offset(self.blocks(self.input(combined)))
                offsets = offsets.transpose(1, 2).reshape(batch, instances, vertices, 2)
                result = result + offsets
            return result

    class ShipContourNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DLA60()
            self.heatmap = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), Mish(),
                                         nn.Conv2d(64, 1, 1))
            self.size = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), Mish(),
                                      nn.Conv2d(64, 2, 1))
            self.offset = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), Mish(),
                                        nn.Conv2d(64, 2, 1))
            self.snake = ContourEvolution()
            nn.init.constant_(self.heatmap[-1].bias, -2.19)

        def forward(self, image, contours=None):
            features = self.backbone(image)
            output = {"heatmap": torch.sigmoid(self.heatmap(features)),
                      "size": self.size(features), "offset": self.offset(features),
                      "features": features}
            if contours is not None:
                output["contours"] = self.snake(features, contours)
            return output

    return ShipContourNet


def _focal_loss(prediction, target):
    import torch
    positive = target.eq(1).float()
    negative = target.lt(1).float()
    negative_weight = (1 - target).pow(4)
    prediction = prediction.clamp(1e-6, 1 - 1e-6)
    positive_loss = -(prediction.log()) * (1 - prediction).pow(2) * positive
    negative_loss = -((1 - prediction).log()) * prediction.pow(2) * negative_weight * negative
    count = positive.sum()
    return ((positive_loss.sum() + negative_loss.sum()) / count
            if count > 0 else negative_loss.sum())


class ShipContourDataset:
    def __init__(self, annotation_file: str, image_dir: str,
                 image_size: int = 512, max_instances: int = 32):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.max_instances = max_instances
        payload = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        self.images = payload.get("images", [])
        self.annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            self.annotations[int(annotation["image_id"])].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import torch
        info = self.images[index]
        with Image.open(self.image_dir / info["file_name"]) as source:
            resized = source.convert("RGB").resize((self.image_size, self.image_size),
                                                   Image.Resampling.BILINEAR)
        image = np.asarray(resized).astype(np.float32) / 255.
        output_size = self.image_size // 4
        heatmap = np.zeros((output_size, output_size), dtype=np.float32)
        sizes = np.zeros((self.max_instances, 2), dtype=np.float32)
        offsets = np.zeros((self.max_instances, 2), dtype=np.float32)
        indices = np.zeros(self.max_instances, dtype=np.int64)
        valid = np.zeros(self.max_instances, dtype=np.float32)
        initial = np.zeros((self.max_instances, N_VERTICES, 2), dtype=np.float32)
        targets = np.zeros_like(initial)
        annotations = self.annotations.get(int(info["id"]), [])[:self.max_instances]
        scale_x = self.image_size / info["width"] / 4
        scale_y = self.image_size / info["height"] / 4
        for slot, annotation in enumerate(annotations):
            x, y, width, height = map(float, annotation["bbox"])
            box = np.array([x * scale_x, y * scale_y,
                            (x + width) * scale_x, (y + height) * scale_y])
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            integer = np.floor(center).astype(int)
            if not (0 <= integer[0] < output_size and 0 <= integer[1] < output_size):
                continue
            draw_gaussian(heatmap, integer[0], integer[1],
                          gaussian_radius(box[3] - box[1], box[2] - box[0]))
            sizes[slot] = [box[2] - box[0], box[3] - box[1]]
            offsets[slot] = center - integer
            indices[slot] = integer[1] * output_size + integer[0]
            valid[slot] = 1
            initial[slot] = resample_contour(box_to_octagon(box))
            segmentation = annotation.get("segmentation", [])
            if isinstance(segmentation, list) and segmentation:
                longest = max(segmentation, key=len)
                polygon = np.asarray(longest, dtype=np.float32).reshape(-1, 2)
                polygon[:, 0] *= scale_x; polygon[:, 1] *= scale_y
                targets[slot] = resample_contour(polygon)
            else:
                targets[slot] = initial[slot]
        return {"image": torch.from_numpy(image).permute(2, 0, 1),
                "heatmap": torch.from_numpy(heatmap[None]),
                "sizes": torch.from_numpy(sizes), "offsets": torch.from_numpy(offsets),
                "indices": torch.from_numpy(indices), "valid": torch.from_numpy(valid),
                "initial": torch.from_numpy(initial), "targets": torch.from_numpy(targets)}


class ShipContour(ModelAdapter):
    name = "ship_contour"

    def __init__(self, cfg, dataset: str = "ssdd") -> None:
        super().__init__(cfg, dataset)
        self.ckpt = str(Path(self.run_dir) / "ship_contour_best.pt")

    def _record(self):
        if not self.arch_record:
            self.record_architecture({"backbone": "DLA-60", "detector": "CenterNet",
                                      "mask_head": "Deep Snake", "attention": "ECA",
                                      "activation": "Mish"})

    def train(self, data: dict) -> None:
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from .. import clearml_utils as tracking
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_network()().to(device)
        self._record()
        dataset = ShipContourDataset(data["coco_train"], data["train_image_dir"],
                                     self.cfg.img_size)
        recipe = self.training_recipe()
        loader = DataLoader(dataset, batch_size=recipe["batch_size"], shuffle=True,
                            num_workers=self.cfg.workers)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(recipe.get("lr", 1e-4)),
            betas=(float(recipe.get("momentum", .9)), .999))
        best = float("inf")
        for epoch in range(int(recipe["epochs"])):
            self.model.train(); total = steps = 0
            for batch in loader:
                images = batch["image"].to(device)
                initial = batch["initial"].to(device)
                output = self.model(images, initial)
                heatmap_loss = _focal_loss(output["heatmap"], batch["heatmap"].to(device))
                flat_size = output["size"].flatten(2).transpose(1, 2)
                flat_offset = output["offset"].flatten(2).transpose(1, 2)
                gather = batch["indices"].to(device).unsqueeze(-1).expand(-1, -1, 2)
                mask = batch["valid"].to(device).unsqueeze(-1)
                size_loss = (F.l1_loss(flat_size.gather(1, gather),
                                       batch["sizes"].to(device), reduction="none") * mask).sum()
                offset_loss = (F.l1_loss(flat_offset.gather(1, gather),
                                         batch["offsets"].to(device), reduction="none") * mask).sum()
                contour_loss = (F.smooth_l1_loss(output["contours"],
                                                 batch["targets"].to(device),
                                                 reduction="none") *
                                mask.unsqueeze(-1)).sum()
                denominator = mask.sum().clamp_min(1)
                objective = heatmap_loss + (size_loss + offset_loss) / denominator + contour_loss / denominator
                optimizer.zero_grad(set_to_none=True); objective.backward(); optimizer.step()
                total += float(objective.detach()); steps += 1
            average = total / max(steps, 1)
            tracking.log_scalar(f"{self.dataset}/{self.name}", "train_loss", average, epoch + 1)
            if average < best:
                best = average; torch.save(self.model.state_dict(), self.ckpt)

    def load(self, data: dict) -> None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not Path(self.ckpt).is_file():
            raise FileNotFoundError(f"Ship Contour checkpoint not found: {self.ckpt}")
        self.model = _build_network()().to(device)
        self.model.load_state_dict(torch.load(self.ckpt, map_location=device))
        self.model.eval(); self._record()

    @staticmethod
    def _decode(output, top_k=40, threshold=.1):
        import torch
        heatmap = output["heatmap"][:, 0]
        batch, height, width = heatmap.shape
        scores, indices = torch.topk(heatmap.reshape(batch, -1), min(top_k, height * width))
        ys = (indices // width).float(); xs = (indices % width).float()
        offsets = output["offset"].flatten(2).transpose(1, 2)
        sizes = output["size"].flatten(2).transpose(1, 2).clamp_min(0)
        gather = indices.unsqueeze(-1).expand(-1, -1, 2)
        offsets = offsets.gather(1, gather); sizes = sizes.gather(1, gather)
        centers = torch.stack((xs, ys), dim=-1) + offsets
        boxes = torch.cat((centers - sizes / 2, centers + sizes / 2), dim=-1)
        return [(boxes[item][scores[item] >= threshold], scores[item][scores[item] >= threshold])
                for item in range(batch)]

    def _predict_native(self, image_path: str):
        import torch
        if self.model is None:
            raise RuntimeError("Ship Contour is not loaded")
        device = next(self.model.parameters()).device
        with Image.open(image_path) as source:
            original = source.convert("RGB"); width, height = original.size
            resized = original.resize((self.cfg.img_size, self.cfg.img_size), Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
        tensor = tensor.div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(tensor)
            decoded = self._decode(output, self.cfg.max_dets, self.cfg.score_threshold)[0]
            boxes, scores = decoded
            if len(boxes) == 0:
                return [], []
            initial = np.stack([resample_contour(box_to_octagon(box.cpu().numpy()))
                                for box in boxes])
            contours = torch.from_numpy(initial).float().unsqueeze(0).to(device)
            evolved = self.model.snake(output["features"], contours)[0].cpu().numpy()
        masks = []
        for contour in evolved:
            contour[:, 0] *= width / (self.cfg.img_size / 4)
            contour[:, 1] *= height / (self.cfg.img_size / 4)
            canvas = Image.new("1", (width, height), 0)
            ImageDraw.Draw(canvas).polygon([tuple(point) for point in contour], fill=1)
            masks.append(np.asarray(canvas, dtype=np.uint8))
        return masks, scores.cpu().numpy().tolist()

    def predict_coco(self, data: dict) -> list[dict]:
        payload = json.loads(Path(data["coco_test"]).read_text(encoding="utf-8"))
        images = payload.get("images", [])
        state_path = Path(self.run_dir) / "evaluation_progress.json"
        output: list[dict] = []
        completed: set[int] = set()
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                output = state.get("predictions", [])
                completed = {int(item) for item in
                             state.get("completed_image_ids", [])}
                print(f"Resuming {self.dataset}/{self.name} evaluation: "
                      f"{len(completed)}/{len(images)} images already complete")
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                print(f"Ignoring unreadable evaluation state: {state_path}")
                output, completed = [], set()

        remaining = [info for info in images if int(info["id"]) not in completed]
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(remaining, total=len(images),
                            initial=min(len(completed), len(images)),
                            desc=f"Evaluating {self.dataset}/{self.name}", unit="image")
        except ImportError:
            iterator = remaining

        processed_since_save = 0
        for info in iterator:
            image_id = int(info["id"])
            masks, scores = self._predict_native(
                str(Path(data["test_image_dir"]) / info["file_name"]))
            output.extend(detection(image_id, mask, float(score))
                          for mask, score in zip(masks, scores) if mask.any())
            completed.add(image_id)
            processed_since_save += 1
            if processed_since_save >= 25:
                self._save_evaluation_state(state_path, completed, output)
                processed_since_save = 0

        self._save_evaluation_state(state_path, completed, output)
        final_path = Path(self.run_dir) / "predictions.json"
        temporary = final_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(output), encoding="utf-8")
        temporary.replace(final_path)
        print(f"Predictions saved: {final_path}")
        return output

    @staticmethod
    def _save_evaluation_state(path: Path, completed: set[int],
                               predictions: list[dict]) -> None:
        state = {"completed_image_ids": sorted(completed),
                 "predictions": predictions}
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state), encoding="utf-8")
        temporary.replace(path)

    def predict_one(self, image_path: str):
        return self._predict_native(image_path)

    def hyperparams(self) -> dict:
        return {**super().hyperparams(), "backbone": "DLA-60+ECA+Mish",
                "detector": "CenterNet", "contour": "Deep Snake",
                "snake_vertices": N_VERTICES, "snake_iterations": 3}

