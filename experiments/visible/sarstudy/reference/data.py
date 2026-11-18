"""Dataset discovery and deterministic export for the benchmark.

All source formats are converted to one small dictionary contract before any
model sees them. This keeps split selection independent from model code.
"""
from __future__ import annotations

import json
import random
import shutil
import xml.etree.ElementTree as ET
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

CATEGORY = {"id": 1, "name": "ship", "supercategory": "ship"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def is_ssdd_test_stem(stem: str) -> bool:
    if not stem:
        raise ValueError("SSDD filename stem cannot be empty")
    return stem[-1] in {"1", "9"}


def _stable_id(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def _image_for_stem(folder: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _labelme_record(annotation_path: Path) -> dict[str, Any] | None:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_path = _image_for_stem(annotation_path.parent, annotation_path.stem)
    if image_path is None:
        declared = payload.get("imagePath")
        if declared:
            candidate = annotation_path.parent / Path(declared).name
            image_path = candidate if candidate.is_file() else None
    if image_path is None:
        return None
    width = payload.get("imageWidth")
    height = payload.get("imageHeight")
    if not width or not height:
        with Image.open(image_path) as image:
            width, height = image.size
    polygons = []
    for shape in payload.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float64)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            polygons.append(points)
    return {"stem": annotation_path.stem, "image_path": str(image_path),
            "width": int(width), "height": int(height), "polygons": polygons}


def _load_labelme_folder(folder: Path) -> list[dict[str, Any]]:
    records = []
    for annotation_path in sorted(folder.glob("*.json")):
        try:
            record = _labelme_record(annotation_path)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid LabelMe annotation: {annotation_path}") from error
        if record is not None:
            records.append(record)
    return records


def load_ssdd_index(ssdd_dir: str) -> tuple[list[dict], list[dict]]:
    records = _load_labelme_folder(Path(ssdd_dir))
    if not records:
        raise FileNotFoundError(f"No SSDD image/LabelMe pairs found in {ssdd_dir}")
    training, testing = [], []
    for record in records:
        (testing if is_ssdd_test_stem(record["stem"]) else training).append(record)
    return training, testing


def _decode_segmentation(segmentation: Any) -> list[np.ndarray]:
    if isinstance(segmentation, list):
        decoded = []
        for flat in segmentation:
            if isinstance(flat, list) and len(flat) >= 6 and len(flat) % 2 == 0:
                decoded.append(np.asarray(flat, dtype=np.float64).reshape(-1, 2))
        return decoded
    if isinstance(segmentation, dict):
        try:
            import cv2
            from pycocotools import mask as mask_utils
        except ImportError as error:
            raise ImportError("RLE annotations require pycocotools and OpenCV") from error
        mask = mask_utils.decode(segmentation).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return [item.reshape(-1, 2).astype(np.float64)
                for item in contours if len(item) >= 3]
    return []


def _coco_records(annotation_file: Path, image_folder: Path) -> list[dict[str, Any]]:
    if not annotation_file.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {annotation_file}")
    payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        grouped[int(annotation["image_id"])].append(annotation)
    records = []
    for info in sorted(payload.get("images", []), key=lambda item: int(item["id"])):
        image_path = image_folder / info["file_name"]
        if not image_path.is_file():
            alternate = _find_image(Path(annotation_file).parent.parent,
                                    Path(info["file_name"]).stem)
            image_path = alternate or image_path
        polygons, boxes = [], []
        for annotation in grouped.get(int(info["id"]), []):
            segmentation = annotation.get('segmentation')
            parts = _decode_segmentation(segmentation)
            if isinstance(segmentation,dict) or len(parts)!=1 or annotation.get('iscrowd',0):
                raise ValueError('Expected one non-crowd polygon per ship instance. '
                                 'RLE/multipart annotations need an explicitly reviewed conversion; '
                                 f'annotation={annotation.get("id")} file={annotation_file}')
            polygons.extend(parts)
            if len(annotation.get("bbox", [])) == 4:
                x, y, width, height = map(float, annotation["bbox"])
                boxes.append([x, y, x + width, y + height])
        records.append({"stem": Path(info["file_name"]).stem,
                        "image_path": str(image_path),
                        "width": int(info["width"]), "height": int(info["height"]),
                        "polygons": polygons, "boxes_orig": boxes,
                        "source_image_id": int(info["id"])})
    return records


def _find_image(folder: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        matches = list(folder.rglob(f"{stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def _preferred_image_dir(root: Path) -> Path:
    for name in ("images", "JPEGImages", "imgs"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root


def load_hrsid_index(hrsid_dir: str) -> tuple[list[dict], list[dict]]:
    root = Path(hrsid_dir)
    images = _preferred_image_dir(root)
    return (_coco_records(root / "annotations" / "train2017.json", images),
            _coco_records(root / "annotations" / "test2017.json", images))


def _nested_release_folder(folder: Path) -> Path:
    nested = folder / folder.name
    return nested if nested.is_dir() else folder


def load_lsssdd_index(root: str, split: str = "test",
                      only_with_ships: bool = True) -> list[dict]:
    dataset_root = Path(root)
    identifiers_file = dataset_root / f"{split}.txt"
    if not identifiers_file.is_file():
        raise FileNotFoundError(f"LS-SSDD split file not found: {identifiers_file}")
    annotations = _nested_release_folder(dataset_root / "Annotations_sub")
    subset = "train" if split == "train" else "test"
    images = _nested_release_folder(dataset_root / f"JPEGImages_sub_{subset}")
    identifiers = [line.strip() for line in
                   identifiers_file.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    records = []
    for stem in identifiers:
        xml_file = annotations / f"{stem}.xml"
        image_file = _image_for_stem(images, stem)
        if not xml_file.is_file() or image_file is None:
            continue
        tree = ET.parse(xml_file)
        boxes = []
        for item in tree.findall("object"):
            box = item.find("bndbox")
            if box is None:
                continue
            boxes.append([float(box.findtext("xmin", "0")),
                          float(box.findtext("ymin", "0")),
                          float(box.findtext("xmax", "0")),
                          float(box.findtext("ymax", "0"))])
        if only_with_ships and not boxes:
            continue
        size = tree.find("size")
        if size is None:
            with Image.open(image_file) as image:
                width, height = image.size
        else:
            width = int(size.findtext("width", "0"))
            height = int(size.findtext("height", "0"))
        records.append({"stem": stem, "image_path": str(image_file),
                        "width": width, "height": height,
                        "polygons": [], "boxes_orig": boxes})
    return records


def _seeded_split(records: list[dict], fraction: float,
                  seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < fraction < 1:
        raise ValueError("train_frac must be between zero and one")
    shuffled = sorted(records, key=lambda item: item["stem"])
    random.Random(seed).shuffle(shuffled)
    boundary = int(len(shuffled) * fraction)
    return shuffled[:boundary], shuffled[boundary:]


def load_visible_index(root: str, train_frac: float = .8,
                       seed: int = 0) -> tuple[list[dict], list[dict]]:
    dataset_root = Path(root)
    annotation_dir = dataset_root / "annotations"
    if annotation_dir.is_dir():
        train_files = sorted(annotation_dir.glob("train*.json"))
        test_files = sorted(annotation_dir.glob("test*.json"))
        val_files = sorted(annotation_dir.glob("val*.json"))
        if train_files and (test_files or val_files):
            image_dir = _preferred_image_dir(dataset_root)
            return (_coco_records(train_files[0], image_dir),
                    _coco_records((test_files or val_files)[0], image_dir))
        coco_files = sorted(annotation_dir.glob("*.json"))
        if len(coco_files) == 1:
            records = _coco_records(coco_files[0], _preferred_image_dir(dataset_root))
            return _seeded_split(records, train_frac, seed)
    records = _load_labelme_folder(dataset_root)
    if records:
        return _seeded_split(records, train_frac, seed)
    raise FileNotFoundError(f"No supported visible-light annotations in {root}")


def assert_split(name: str, train: list[dict], test: list[dict],
                 expected_train: int | None = None,
                 expected_test: int | None = None) -> None:
    if not train or not test:
        raise ValueError(f"{name} split is empty: train={len(train)}, test={len(test)}")
    if expected_train is not None and len(train) != expected_train:
        raise ValueError(f"{name} train split: expected {expected_train}, got {len(train)}")
    if expected_test is not None and len(test) != expected_test:
        raise ValueError(f"{name} test split: expected {expected_test}, got {len(test)}")
    overlap = {item["stem"] for item in train} & {item["stem"] for item in test}
    if overlap:
        raise ValueError(f"{name} split leakage detected for {len(overlap)} images")


def _polygon_stats(polygon: np.ndarray, width: int,
                   height: int) -> tuple[float, list[float]]:
    clipped = np.asarray(polygon, dtype=np.float64).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    x, y = clipped[:, 0], clipped[:, 1]
    area = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2
    left, top = float(x.min()), float(y.min())
    return area, [left, top, float(x.max() - left), float(y.max() - top)]


def records_to_coco(records: list[dict], out_path: str) -> dict:
    images, annotations = [], []
    annotation_id = 1
    for record in records:
        image_id = int(record.get("source_image_id") or _stable_id(record["stem"]))
        images.append({"id": image_id,
                       "file_name": Path(record["image_path"]).name,
                       "width": int(record["width"]),
                       "height": int(record["height"])})
        for polygon in record.get("polygons", []):
            points = np.asarray(polygon, dtype=np.float64)
            if points.shape[0] < 3:
                continue
            area, bbox = _polygon_stats(points, record["width"], record["height"])
            if area <= 0:
                continue
            annotations.append({"id": annotation_id, "image_id": image_id,
                                "category_id": 1,
                                "segmentation": [points.reshape(-1).tolist()],
                                "area": area, "bbox": bbox, "iscrowd": 0})
            annotation_id += 1
    payload = {"info": {"description": "SAR ship segmentation study"},
               "licenses": [], "images": images, "annotations": annotations,
               "categories": [CATEGORY]}
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def boxes_from_records(records: Iterable[dict]) -> dict[int, np.ndarray]:
    result = {}
    for record in records:
        image_id = int(record.get("source_image_id") or _stable_id(record["stem"]))
        result[image_id] = np.asarray(record.get("boxes_orig", []),
                                      dtype=np.float32).reshape(-1, 4)
    return result


def records_to_yolo(records: list[dict], img_out: str, lbl_out: str,
                    copy_images: bool = True) -> None:
    image_dir, label_dir = Path(img_out), Path(lbl_out)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = Path(record["image_path"])
        destination = image_dir / source.name
        if copy_images and (not destination.exists() or
                            destination.stat().st_size != source.stat().st_size):
            shutil.copy2(source, destination)
        lines = []
        width, height = float(record["width"]), float(record["height"])
        for polygon in record.get("polygons", []):
            points = np.asarray(polygon, dtype=np.float64)
            if points.shape[0] < 3:
                continue
            normalised = points / np.asarray([width, height])
            normalised = np.clip(normalised, 0, 1).reshape(-1)
            lines.append("0 " + " ".join(f"{value:.8f}" for value in normalised))
        (label_dir / f"{source.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yolo_yaml(yolo_dir: str) -> str:
    root = Path(yolo_dir).resolve()
    target = root / "data.yaml"
    target.write_text(f"path: {root}\ntrain: images/train\nval: images/val\n"
                      "names:\n  0: ship\n", encoding="utf-8")
    return str(target)


def _load_train_test(name: str, cfg) -> tuple[list[dict], list[dict]]:
    if name == "ssdd":
        return load_ssdd_index(cfg.ssdd_dir)
    if name == "hrsid":
        return load_hrsid_index(cfg.hrsid_dir)
    if name == "visible":
        return load_visible_index(cfg.visible_dir, seed=cfg.seed)
    raise ValueError(f"Dataset {name!r} cannot be used for training")


def _instance_bin(record: dict) -> int:
    count = len(record.get("polygons", [])) or len(record.get("boxes_orig", []))
    return 0 if count == 0 else 1 if count == 1 else 2 if count <= 3 else 3


def stratified_subset(records: list[dict], limit: int | None,
                      seed: int) -> list[dict]:
    """Select a reproducible ship-count-stratified subset."""
    if limit is None or limit >= len(records):
        return list(records)
    if limit < 1:
        raise ValueError("Subset limits must be positive")
    groups: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        groups[_instance_bin(record)].append(record)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda item: item["stem"])
        rng.shuffle(values)
    exact = {key: limit * len(values) / len(records)
             for key, values in groups.items()}
    allocations = {key: min(len(groups[key]), int(value))
                   for key, value in exact.items()}
    remaining = limit - sum(allocations.values())
    order = sorted(groups, key=lambda key: exact[key] - allocations[key], reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    selected = [record for key in sorted(groups)
                for record in groups[key][:allocations[key]]]
    return sorted(selected, key=lambda item: item["stem"])


def _manifest_from_prepared(name: str, cfg) -> dict | None:
    train_json = Path(cfg.coco_dir(name)) / "train.json"
    test_json = Path(cfg.coco_dir(name)) / "test.json"
    yolo_root = Path(cfg.yolo_dir(name))
    train_images = yolo_root / "images" / "train"
    test_images = yolo_root / "images" / "val"
    if not all(path.exists() for path in
               (train_json, test_json, train_images, test_images)):
        return None
    train = json.loads(train_json.read_text(encoding="utf-8"))
    test = json.loads(test_json.read_text(encoding="utf-8"))
    for payload, path in ((train, train_json), (test, test_json)):
        changed = "info" not in payload or "licenses" not in payload
        payload.setdefault("info", {"description": "Honours SAR ship benchmark"})
        payload.setdefault("licenses", [])
        if changed:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    expected_train = cfg.subset_size(name, "train")
    expected_test = cfg.subset_size(name, "test")
    if expected_train is not None and len(train.get("images", [])) != expected_train:
        return None
    if expected_test is not None and len(test.get("images", [])) != expected_test:
        return None
    return {"dataset": name, "domain": cfg.domain_of(name),
            "source_dir": cfg.dataset_dir(name),
            "train_images": len(train.get("images", [])),
            "test_images": len(test.get("images", [])),
            "train_instances": len(train.get("annotations", [])),
            "test_instances": len(test.get("annotations", [])),
            "coco_train": str(train_json), "coco_test": str(test_json),
            "yolo_yaml": write_yolo_yaml(str(yolo_root)),
            "train_image_dir": str(train_images),
            "test_image_dir": str(test_images), "cache_reused": True}


def prepare_dataset(name: str, cfg) -> dict:
    cfg.ensure_dirs()
    if cfg.reuse_prepared_data:
        cached = _manifest_from_prepared(name, cfg)
        if cached is not None:
            print(f"Reusing prepared {name} data")
            return cached
    train, test = _load_train_test(name, cfg)
    expected = cfg.expected(name)
    assert_split(name, train, test, expected.get("train"), expected.get("test"))
    train = stratified_subset(train, cfg.subset_size(name, "train"), cfg.seed)
    test = stratified_subset(test, cfg.subset_size(name, "test"), cfg.seed + 1)
    train_json = str(Path(cfg.coco_dir(name)) / "train.json")
    test_json = str(Path(cfg.coco_dir(name)) / "test.json")
    train_coco = records_to_coco(train, train_json)
    test_coco = records_to_coco(test, test_json)
    yolo_root = Path(cfg.yolo_dir(name))
    train_images = yolo_root / "images" / "train"
    test_images = yolo_root / "images" / "val"
    records_to_yolo(train, str(train_images), str(yolo_root / "labels" / "train"))
    records_to_yolo(test, str(test_images), str(yolo_root / "labels" / "val"))
    selection = {"dataset": name, "seed": cfg.seed,
                 "train": [item["stem"] for item in train],
                 "test": [item["stem"] for item in test]}
    selection_path = Path(cfg.coco_dir(name)) / "selected_ids.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    return {"dataset": name, "domain": cfg.domain_of(name),
            "source_dir": cfg.dataset_dir(name),
            "train_images": len(train), "test_images": len(test),
            "train_instances": len(train_coco["annotations"]),
            "test_instances": len(test_coco["annotations"]),
            "coco_train": train_json, "coco_test": test_json,
            "yolo_yaml": write_yolo_yaml(str(yolo_root)),
            "train_image_dir": str(train_images),
            "test_image_dir": str(test_images),
            "selection_file": str(selection_path), "cache_reused": False}


def prepare_generalization_data(cfg) -> dict:
    if cfg.generalization_dataset != "ls_ssdd":
        raise ValueError(f"Unsupported generalization dataset: {cfg.generalization_dataset}")
    records = load_lsssdd_index(cfg.ls_ssdd_dir, "test",
                                cfg.generalization_only_with_ships)
    if cfg.generalization_max_images is not None:
        records = stratified_subset(records, cfg.generalization_max_images,
                                    cfg.seed + 2)
    local_images = Path(cfg.yolo_dir("ls_ssdd")) / "images" / "test"
    local_images.mkdir(parents=True, exist_ok=True)
    for record in records:
        source = Path(record["image_path"])
        destination = local_images / source.name
        if not destination.exists() or destination.stat().st_size != source.stat().st_size:
            shutil.copy2(source, destination)
        record["image_path"] = str(destination)
    output = str(Path(cfg.coco_dir("ls_ssdd")) / "test_boxes_only.json")
    records_to_coco(records, output)
    return {"dataset": "ls_ssdd", "domain": "sar",
            "source_dir": cfg.ls_ssdd_dir, "test_images": len(records),
            "test_boxes": sum(len(item.get("boxes_orig", [])) for item in records),
            "coco_test": output,
            "test_image_dir": str(local_images) if records else "",
            "gt_boxes": boxes_from_records(records), "has_masks": False}


def prepare_data(cfg) -> dict[str, dict]:
    cfg.ensure_dirs()
    return {name: prepare_dataset(name, cfg) for name in cfg.train_datasets}
