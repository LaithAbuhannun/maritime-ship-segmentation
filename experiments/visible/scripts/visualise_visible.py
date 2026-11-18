"""Draw the same held-out test image for each completed model, from saved masks."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs/visible.json').read_text())
    manifest = json.loads((root / 'data/visible/manifest.json').read_text())
    # Fixed examples, never selected using a model's predictions or test AP.
    record = manifest['records']['test'][0]
    image = Image.open(root / 'data/visible' / record['relative_path']).convert('RGB')
    output = root / 'reports/qualitative'; output.mkdir(parents=True, exist_ok=True)
    panels = []
    original = np.asarray(image)
    truth = image.copy()
    draw = ImageDraw.Draw(truth)
    for polygon in record['polygons']:
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        draw.line(points + [points[0]], fill=(0, 255, 0), width=3)
    panels.extend([('Input image', original), ('Ground truth', np.asarray(truth))])
    for model in cfg['models']:
        folder = root / 'runs/visible' / model / cfg['experiment']
        prediction = folder / 'predictions/test.json'
        if not (folder / 'evaluation_complete.json').is_file() or not prediction.is_file():
            print('No completed results:', model)
            continue
        threshold = json.loads((folder / 'threshold.json').read_text())['value']
        masks = [p for p in json.loads(prediction.read_text())
                 if int(p['image_id']) == int(record['source_image_id']) and p['score'] >= threshold]
        overlay = original.copy().astype(float)
        for prediction in masks:
            rle = dict(prediction['segmentation'])
            if isinstance(rle['counts'], str): rle['counts'] = rle['counts'].encode('ascii')
            mask = mask_utils.decode(rle).astype(bool)
            if mask.shape != original.shape[:2]: raise ValueError('Prediction shape mismatch')
            overlay[mask] = .55 * overlay[mask] + .45 * np.array([255, 0, 0])
        panel = np.clip(overlay, 0, 255).astype(np.uint8)
        label = model + (' (GT-box prompted)' if model == 'samsar' else '')
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, title, content in zip(axes, ['Input', 'Ground truth', label],
                                      [original, np.asarray(truth), panel]):
            ax.imshow(content); ax.set_title(title); ax.axis('off')
        fig.tight_layout(); fig.savefig(output / f'{model}.png', dpi=180)
        plt.close(fig)
        panels.append((label, panel))
    columns = 3
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
    for ax in axes.flat: ax.axis('off')
    for ax, (label, content) in zip(axes.flat, panels):
        ax.imshow(content); ax.set_title(label)
    fig.tight_layout(); fig.savefig(output / 'all_models_same_test_image.png', dpi=180)
    plt.close(fig)
    print('Saved qualitative figures:', output)


if __name__ == '__main__': main()
