"""Create one immutable train/validation/test split per experiment/dataset."""
from pathlib import Path
import json
import random
import shutil
import numpy as np
from .storage import atomic_json, file_hash, fingerprint, read_json, require_identity
from .reference.config import Config
from .reference.data import (_load_train_test, stratified_subset, records_to_coco,
                             records_to_yolo, load_lsssdd_index, boxes_from_records)

def split_training_pool(pool, fraction, seed):
    if len(pool) < 2:
        raise ValueError("Need at least two training-pool images")
    # Duplicate-content images are an indivisible group. Keeping them together
    # prevents identical pixels appearing in both training and validation.
    groups = {}
    for record in sorted(pool, key=lambda r: r["stem"]):
        groups.setdefault(record["sha256"], []).append(record)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    nval = max(1, min(len(pool)-1, round(len(pool)*fraction)))
    val, train = [], []
    # Prefer the requested count without breaking a duplicate group. Singleton
    # groups fill any remaining slots after larger groups are considered.
    for group in sorted(ordered, key=lambda g: len(g), reverse=True):
        if len(val) + len(group) <= nval:
            val.extend(group)
        else:
            train.extend(group)
    if not train or not val:
        raise ValueError("Could not form non-empty grouped train/validation splits")
    return train, val

def check_disjoint(groups):
    seen_ids, seen_files, seen_hashes = {}, {}, {}
    for split, records in groups.items():
        if not records:
            raise ValueError(f"Empty {split} split")
        local_ids, local_files = set(), set()
        for record in records:
            identity = record["stem"]
            filename = Path(record["image_path"]).name
            if identity in local_ids or filename in local_files:
                raise ValueError(f"Duplicate identity/filename within {split}: {identity}")
            local_ids.add(identity); local_files.add(filename)
            for key, mapping in ((identity, seen_ids), (filename, seen_files),
                                  (record.get("sha256"), seen_hashes)):
                if key is None:
                    continue
                if key in mapping and mapping[key] != split:
                    raise ValueError(f"Leakage/duplicate between {mapping[key]} and {split}: {identity}")
                mapping[key] = split

def exclude_test_duplicates(pool, test):
    """Preserve test and remove any overlapping record from the training pool."""
    test_ids = {record["stem"] for record in test}
    test_files = {Path(record["image_path"]).name for record in test}
    test_hashes = {record["sha256"] for record in test}
    clean, excluded = [], []
    for record in sorted(pool, key=lambda r: r["stem"]):
        reasons = []
        if record["stem"] in test_ids:
            reasons.append("identity")
        if Path(record["image_path"]).name in test_files:
            reasons.append("filename")
        if record["sha256"] in test_hashes:
            reasons.append("sha256")
        if reasons:
            excluded.append({"stem": record["stem"],
                             "image_path": record["image_path"],
                             "sha256": record["sha256"],
                             "reasons": reasons})
        else:
            clean.append(record)
    return clean, excluded

def _serialise(record):
    return {k: v.tolist() if isinstance(v, np.ndarray) else
            [x.tolist() if isinstance(x, np.ndarray) else x for x in v] if isinstance(v, list)
            else v for k, v in record.items()}

def _baseline_cfg(cfg):
    root = Path(cfg.data_root)
    return Config(ssdd_dir=str(root/'SSDD_coco'), hrsid_dir=str(root/'HRSID'),
                  ls_ssdd_dir=str(root/'LS-SSD'))

def prepare(cfg, dataset):
    if dataset not in {"ssdd", "hrsid"}:
        raise ValueError(dataset)
    identity = {"dataset": dataset, "seed": cfg.seed,
                "validation_fraction": cfg.validation_fraction,
                "pool_limit": cfg.hrsid_pool if dataset == "hrsid" else None,
                "test_limit": cfg.hrsid_test if dataset == "hrsid" else None,
                "data_root": str(Path(cfg.data_root).resolve()), "smoke":cfg.smoke, "schema": 4}
    folder = Path(cfg.project_root)/'splits'/cfg.experiment/dataset
    require_identity(folder/'identity.json', identity)
    manifest_path = folder/'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        groups = manifest['records']
        if fingerprint(groups) != manifest['fingerprint']:
            raise ValueError("Split manifest changed or corrupted")
    else:
        pool, test = _load_train_test(dataset, _baseline_cfg(cfg))
        if identity['pool_limit'] is not None:
            pool = stratified_subset(pool, identity['pool_limit'], cfg.seed)
        if identity['test_limit'] is not None:
            test = stratified_subset(test, identity['test_limit'], cfg.seed+1)
        print(f"Hashing {dataset} source images before grouped splitting", flush=True)
        for record in pool + test:
            record['sha256'] = file_hash(record['image_path'])
        # The published test set remains fixed. Any duplicate copy in the
        # training pool is excluded before the seeded train/validation split.
        pool, exclusions = exclude_test_duplicates(pool, test)
        atomic_json(folder/'leakage_audit.json', {
            "dataset": dataset, "seed": cfg.seed,
            "policy": "preserve_test_exclude_overlapping_pool_records",
            "excluded_count": len(exclusions),
            "excluded_pool_records": exclusions,
        })
        print(f"Excluded {len(exclusions)} pool image(s) overlapping the test set", flush=True)
        check_disjoint({'pool': pool, 'test': test})
        train, val = split_training_pool(pool, cfg.validation_fraction, cfg.seed)
        groups = {"train": train, "val": val, "test": test}
        if cfg.smoke:
            groups = {s: rs[:8 if s=='train' else 4] for s,rs in groups.items()}
        for split, records in groups.items():
            print(f"Fingerprinting {dataset}/{split}: {len(records)} images", flush=True)
            for r in records:
                if 'sha256' not in r:
                    r['sha256'] = file_hash(r['image_path'])
        groups = {s: [_serialise(r) for r in rs] for s, rs in groups.items()}
        check_disjoint(groups)
        manifest = {"identity": identity, "records": groups, "fingerprint": fingerprint(groups)}
        atomic_json(manifest_path, manifest)
    check_disjoint(groups)
    cache = Path(cfg.cache_root)/manifest['fingerprint']/dataset
    result = {"dataset": dataset, "fingerprint": manifest['fingerprint'],
              "manifest_path": str(manifest_path)}
    for split, records in groups.items():
        img_dir = cache/'images'/split
        img_dir.mkdir(parents=True, exist_ok=True)
        for r in records:
            target = img_dir/Path(r['image_path']).name
            if not target.exists() or file_hash(target) != r['sha256']:
                if file_hash(r['image_path']) != r['sha256']:
                    raise ValueError(f"Source image changed: {r['image_path']}")
                shutil.copy2(r['image_path'], target)
        records_to_yolo(records, str(img_dir), str(cache/'labels'/split), copy_images=False)
        ann = cache/f'{split}.json'
        payload = records_to_coco(records, str(ann))
        ids = [r['id'] for r in payload['images']]
        if len(set(ids)) != len(ids):
            raise ValueError("COCO image ID collision")
        atomic_json(folder/f'{split}.json', payload)
        result[f'coco_{split}'] = str(ann)
        result[f'{split}_image_dir'] = str(img_dir)
        result[f'{split}_images'] = len(records)
    import yaml
    yaml_path = cache/'data.yaml'
    yaml_path.write_text(yaml.safe_dump({"path": str(cache), "train": "images/train",
                          "val": "images/val", "test": "images/test", "names": {0: "ship"}}))
    result['yolo_yaml'] = str(yaml_path)
    print({s: result[f'{s}_images'] for s in ('train','val','test')}, flush=True)
    return result

def evaluation_view(data, split):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    return {**data, 'coco_test': data[f'coco_{split}'],
            'test_image_dir': data[f'{split}_image_dir'], 'evaluation_split': split}

def prepare_generalization(cfg):
    folder = Path(cfg.project_root)/'splits'/cfg.experiment/'ls_ssdd'
    identity = {"seed": cfg.seed+2, "images": cfg.generalization_images,
                "source": str(Path(cfg.data_root)/'LS-SSD'), "only_with_ships": True,
                "smoke":cfg.smoke}
    require_identity(folder/'identity.json', identity)
    path = folder/'manifest.json'
    if path.exists():
        saved = read_json(path)
        records = saved['records']
        if fingerprint(records) != saved['fingerprint']:
            raise ValueError('Generalization manifest changed or corrupted')
    else:
        records = load_lsssdd_index(identity['source'], 'test', True)
        records = stratified_subset(records, cfg.generalization_images, cfg.seed+2)
        if cfg.smoke: records = records[:4]
        records = [_serialise(r) for r in records]
        for r in records:
            r['sha256'] = file_hash(r['image_path'])
        atomic_json(path, {"records": records, 'fingerprint':fingerprint(records)})
    check_disjoint({'generalization': records})
    fp = fingerprint(records)
    local = Path(cfg.cache_root)/fp/'ls_ssdd'
    local.mkdir(parents=True, exist_ok=True)
    for r in records:
        target = local/Path(r['image_path']).name
        if not target.exists() or file_hash(target) != r['sha256']:
            if file_hash(r['image_path']) != r['sha256']:
                raise ValueError("LS-SSD source changed")
            shutil.copy2(r['image_path'], target)
    ann = local/'test.json'
    records_to_coco(records, str(ann))
    return {'dataset':'ls_ssdd', 'evaluation_split':'generalization', 'fingerprint':fp,
            'coco_test':str(ann), 'test_image_dir':str(local),
            'gt_boxes':boxes_from_records(records), 'manifest_path':str(path)}
