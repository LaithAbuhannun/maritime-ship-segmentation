"""Freeze a seeded, source-grouped visible-light subset and package its images."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import zipfile
from PIL import Image


def prepare(source, output, seed=42):
    source, output = Path(source), Path(output)
    items = []
    categories = {}
    for split in ('train', 'valid'):
        payload = json.loads((source / split / '_annotations.coco.json').read_text())
        categories.update({a['id']: a['name'] for a in payload['categories']})
        annotations = defaultdict(list)
        for ann in payload['annotations']:
            if ann.get('iscrowd', 0) or not isinstance(ann.get('segmentation'), list) or len(ann['segmentation']) != 1:
                raise ValueError(f'Unsupported crowd/multipart/RLE instance: {split}/{ann["id"]}')
            if len(ann['segmentation'][0]) < 6 or len(ann['segmentation'][0]) % 2:
                raise ValueError(f'Invalid polygon: {split}/{ann["id"]}')
            annotations[ann['image_id']].append(ann)
        for info in payload['images']:
            path = source / split / info['file_name']
            content = path.read_bytes()
            with Image.open(path) as image:
                rgb = image.convert('RGB')
                if rgb.size != (info['width'], info['height']):
                    raise ValueError(f'Incorrect dimensions: {path}')
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            # Roboflow variants retain the original filename before .rf.<hash>.
            group = re.split(r'\.rf\.', info['file_name'])[0].lower()
            items.append(dict(source_split=split, original_file=info['file_name'],
                source_path=str(path), group=group, sha256=hashlib.sha256(content).hexdigest(),
                pixel_hash=pixel_hash, width=info['width'], height=info['height'],
                annotations=annotations[info['id']]))

    # Connect source-name and decoded-pixel duplicates transitively.
    parents = list(range(len(items)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, item in enumerate(items):
        for key in ('group', 'pixel_hash', 'sha256'):
            identity = (key, item[key])
            if identity in seen:
                parents[find(i)] = find(seen[identity])
            seen[identity] = i
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[find(i)].append(item)
    # Remove identical-pixel copies; preserve the valid representative as held out.
    clean = []
    for group in groups.values():
        unique = {}
        for item in sorted(group, key=lambda a: (a['source_split'] != 'valid', a['original_file'])):
            unique.setdefault(item['pixel_hash'], item)
        rows = list(unique.values())
        canonical = min(a['group'] for a in group)
        for row in rows:
            row['source_group'] = canonical
        clean.append((any(a['source_split'] == 'valid' for a in group), rows))
    pool = [rows for held, rows in clean if not held]
    held = [rows for is_held, rows in clean if is_held]
    rng = random.Random(seed)
    pool.sort(key=lambda g: g[0]['source_group']); held.sort(key=lambda g: g[0]['source_group'])
    rng.shuffle(pool); rng.shuffle(held)
    def take(groups, limit):
        selected, rest = [], []
        for group in groups:
            if sum(map(len, selected)) + len(group) <= limit:
                selected.append(group)
            else:
                rest.append(group)
        return selected, rest
    chosen_pool, _ = take(pool, 1250)
    val_groups, train_groups = take(chosen_pool, 250)
    test_groups, _ = take(held, 500)
    splits = {name: [a for g in rows for a in g] for name, rows in
              [('train', train_groups), ('val', val_groups), ('test', test_groups)]}
    manifest = {'seed': seed, 'class_policy': 'all annotated ship types merged to category 1: ship',
        'selection_policy': 'source-grouped; original valid groups held out for test; pixel duplicates removed',
        'source_counts': {s: sum(x['source_split'] == s for x in items) for s in ('train', 'valid')},
        'source_categories': categories, 'counts': {s: len(a) for s, a in splits.items()}, 'records': {}}
    for split, rows in splits.items():
        manifest['records'][split] = []
        for i, row in enumerate(rows, 1):
            name = f'{split}_{i:05d}.jpg'
            manifest['records'][split].append(dict(stem=Path(name).stem,
                relative_path=f'images/{split}/{name}', source_group=row['source_group'],
                width=row['width'], height=row['height'], sha256=row['sha256'],
                pixel_hash=row['pixel_hash'], source_image_id=i,
                polygons=[a['segmentation'][0] for a in row['annotations']],
                original_file=row['original_file'], original_split=row['source_split']))
    for key in ('source_group', 'pixel_hash', 'sha256'):
        sets = {s: {r[key] for r in rows} for s, rows in manifest['records'].items()}
        assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest['records'], sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('Visible_Ship_Study/data/visible/manifest.json', json.dumps(manifest, indent=2))
        for split, rows in splits.items():
            for row, record in zip(rows, manifest['records'][split]):
                archive.write(row['source_path'], 'Visible_Ship_Study/data/visible/' + record['relative_path'])
    report = output.with_suffix('.audit.json')
    report.write_text(json.dumps({k: v for k, v in manifest.items() if k != 'records'}, indent=2))
    print(json.dumps(manifest['counts']), flush=True)
    print(f'Dataset ZIP: {output} ({output.stat().st_size / 1024**2:.1f} MB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source'); parser.add_argument('output')
    args = parser.parse_args()
    prepare(args.source, args.output)
