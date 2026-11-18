"""Prepare identical frozen visible-light splits for every model."""
from pathlib import Path
import hashlib
import json
import shutil
from .storage import atomic_json, file_hash, fingerprint, read_json, require_identity
from .reference.data import records_to_coco, records_to_yolo


def prepare_visible(cfg):
    root = Path(cfg.project_root)
    source = root / 'data/visible'
    bundled = read_json(source / 'manifest.json')
    if bundled['seed'] != cfg.seed:
        raise ValueError('Dataset seed differs from configuration; rebuild the dataset bundle')
    expected = hashlib.sha256(json.dumps(bundled['records'], sort_keys=True).encode()).hexdigest()
    if bundled['fingerprint'] != expected:
        raise ValueError('Frozen dataset manifest changed')
    groups = {}
    for split, rows in bundled['records'].items():
        records = []
        for row in rows:
            record = {**row, 'image_path': str(source / row['relative_path']),
                      'polygons': [[p[i:i+2] for i in range(0, len(p), 2)] for p in row['polygons']]}
            records.append(record)
        groups[split] = records[:8 if split == 'train' else 4] if cfg.smoke else records
    from .data import check_disjoint
    check_disjoint(groups)
    for key in ('source_group', 'pixel_hash'):
        seen = {}
        for split, rows in groups.items():
            for row in rows:
                if row[key] in seen and seen[row[key]] != split:
                    raise ValueError(f'{key} overlap between splits')
                seen[row[key]] = split
    folder = root / 'splits' / cfg.experiment / 'visible'
    identity = {'bundle_fingerprint': expected, 'seed': cfg.seed, 'smoke': cfg.smoke}
    require_identity(folder / 'identity.json', identity)
    manifest = {'identity': identity, 'records': groups, 'fingerprint': fingerprint(groups)}
    atomic_json(folder / 'manifest.json', manifest)
    atomic_json(folder / 'leakage_audit.json', {'seed': cfg.seed,
        'counts': {s: len(r) for s, r in groups.items()}, 'checks': ['source_group', 'pixel_hash', 'sha256'],
        'class_policy': bundled['class_policy'], 'bundle_fingerprint': expected})
    cache = Path(cfg.cache_root) / manifest['fingerprint'] / 'visible'
    data = {'dataset': 'visible', 'fingerprint': manifest['fingerprint'],
            'manifest_path': str(folder / 'manifest.json')}
    for split, records in groups.items():
        images = cache / 'images' / split
        images.mkdir(parents=True, exist_ok=True)
        for row in records:
            target = images / Path(row['image_path']).name
            if not target.exists() or file_hash(target) != row['sha256']:
                if file_hash(row['image_path']) != row['sha256']:
                    raise ValueError(f'Source image changed: {row["image_path"]}')
                shutil.copy2(row['image_path'], target)
        records_to_yolo(records, str(images), str(cache / 'labels' / split), copy_images=False)
        ann = cache / f'{split}.json'
        payload = records_to_coco(records, str(ann))
        atomic_json(folder / f'{split}.json', payload)
        data[f'coco_{split}'] = str(ann)
        data[f'{split}_image_dir'] = str(images)
        data[f'{split}_images'] = len(records)
    import yaml
    path = cache / 'data.yaml'
    path.write_text(yaml.safe_dump({'path': str(cache), 'train': 'images/train',
        'val': 'images/val', 'test': 'images/test', 'names': {0: 'ship'}}))
    data['yolo_yaml'] = str(path)
    print({s: data[f'{s}_images'] for s in groups}, flush=True)
    return data
