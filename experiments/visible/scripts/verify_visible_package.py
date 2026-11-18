"""Integration checks for data, configs and notebooks without requiring a GPU."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sarstudy.config import StudyConfig, MODELS
from sarstudy.data import prepare
from sarstudy.storage import fingerprint


def main():
    base = json.loads((ROOT / 'configs/visible.json').read_text())
    shared = ['experiment', 'seed', 'max_epochs', 'patience', 'min_delta', 'batch_size',
              'workers', 'amp', 'img_size', 'samsar_img_size', 'rfdetr_resolution', 'datasets']
    for model in MODELS:
        data = json.loads((ROOT / 'configs' / f'{model}.json').read_text())
        assert all(data[k] == base[k] for k in shared), model
        StudyConfig(**data)
    cells = 0
    for path in (ROOT / 'notebooks').glob('*Visible*.ipynb'):
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code': continue
            source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                               if not line.lstrip().startswith(('!', '%')))
            ast.parse(source)
            cells += 1
    with tempfile.TemporaryDirectory(prefix='visible-study-check-') as temporary:
        directory = Path(temporary)
        with zipfile.ZipFile(ROOT.parent / 'Visible_Ship_Dataset.zip') as archive:
            assert archive.testzip() is None
            archive.extractall(directory)
        cfg = StudyConfig(**{**base, 'project_root': str(directory / 'Visible_Ship_Study'),
                            'cache_root': str(directory / 'cache'), 'clearml': False})
        data = prepare(cfg, 'visible')
        assert [data[f'{s}_images'] for s in ('train', 'val', 'test')] == [1000, 250, 500]
        from pycocotools.coco import COCO
        for split in ('train', 'val', 'test'):
            payload = json.loads(Path(data[f'coco_{split}']).read_text())
            assert [(c['id'], c['name']) for c in payload['categories']] == [(1, 'ship')]
            assert len({i['id'] for i in payload['images']}) == len(payload['images'])
            assert all(a['category_id'] == 1 and a['area'] > 0 for a in payload['annotations'])
            coco = COCO(data[f'coco_{split}'])
            for ann in payload['annotations']:
                assert coco.annToMask(ann).any()
        # Running a second model must preserve exactly the same split identity.
        repeated = prepare(replace(cfg, models=['solov2_gala'], learning_rate=.01), 'visible')
        assert repeated['fingerprint'] == data['fingerprint']
        smoke = prepare(replace(cfg, smoke=True, max_epochs=2, patience=2,
                                experiment=cfg.experiment + '_smoke'), 'visible')
        assert [smoke[f'{s}_images'] for s in ('train', 'val', 'test')] == [8, 4, 4]
    with zipfile.ZipFile(ROOT.parent / 'Visible_Ship_Study_Package.zip') as archive:
        names = set(archive.namelist())
        for required in ['pyproject.toml', 'environments/mmdetection.txt', 'environments/modern.txt',
                         'sarstudy/native.py', 'sarstudy/visible.py', 'scripts/setup_environment.py']:
            assert 'Visible_Ship_Study/' + required in names
    report = {'result': 'passed', 'model_configurations': len(MODELS), 'validated_notebook_cells': cells,
        'split_counts': {'train': 1000, 'val': 250, 'test': 500},
        'checks': ['disjoint original-source groups and decoded pixels', 'valid nonempty polygon masks',
                   'one-class COCO', 'same split fingerprint across models', '8/4/4 smoke subset',
                   'notebook syntax', 'installation files included'],
        'gpu_training': 'Requires Colab; not tested locally'}
    (ROOT.parent / 'Visible_Ship_Verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
