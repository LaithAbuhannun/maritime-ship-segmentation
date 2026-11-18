"""Create model configurations and clean Colab notebooks for this study."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['yolo26n_seg', 'yolo26s_seg', 'yolo26m_seg', 'rfdetr_seg',
          'samsar', 'ship_contour', 'swininsseg', 'solov2_gala']
BASE = dict(project_root='/content/drive/MyDrive/Visible_Ship_Study',
    data_root='/content/drive/MyDrive/Visible_Ship_Study/data',
    cache_root='/content/visible_ship_cache', experiment='visible_v1_seed42',
    seed=42, validation_fraction=.2, hrsid_pool=1250, hrsid_test=500,
    generalization_images=500, max_epochs=500, patience=20, min_delta=.1,
    batch_size=1, workers=0, img_size=512, rfdetr_resolution=504,
    samsar_img_size=512, amp=False, learning_rate=.0001,
    prediction_floor=.001, max_dets=100, fps_images=100, fps_warmup=10,
    clearml=True, clearml_project='Visible Ship Segmentation Study',
    samsar_repo='/content/drive/MyDrive/SAMSAR_Implementation',
    sam_checkpoint='/content/drive/MyDrive/Visible_Ship_Study/pretrained/sam_vit_b_01ec64.pth',
    datasets=['visible'], models=MODELS)


def md(text):
    return dict(cell_type='markdown', metadata={}, source=text.splitlines(True))


def code(text):
    return dict(cell_type='code', metadata={}, source=text.splitlines(True),
                outputs=[], execution_count=None)


MOUNT = '''from pathlib import Path
from google.colab import drive
import json, os, sys, subprocess, zipfile
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive')
ROOT = DRIVE / 'Visible_Ship_Study'
package = DRIVE / 'Visible_Ship_Study_Package.zip'
assert package.is_file(), f'Upload {package.name} to MyDrive'
with zipfile.ZipFile(package) as archive:
    assert archive.testzip() is None, 'Package ZIP is damaged'
    archive.extractall(DRIVE)
required = ['pyproject.toml', 'environments/modern.txt', 'environments/mmdetection.txt',
            'scripts/setup_environment.py', 'sarstudy/visible.py', 'configs/visible.json']
for name in required:
    assert (ROOT / name).is_file(), f'Package file missing: {name}'
manifest = ROOT / 'data/visible/manifest.json'
if not manifest.is_file():
    dataset_zip = DRIVE / 'Visible_Ship_Dataset.zip'
    assert dataset_zip.is_file(), f'Upload {dataset_zip.name} to MyDrive'
    with zipfile.ZipFile(dataset_zip) as archive:
        assert archive.testzip() is None, 'Dataset ZIP is damaged'
        archive.extractall(DRIVE)
dataset = json.loads(manifest.read_text())
assert dataset['seed'] == 42
assert dataset['counts'] == {'train': 1000, 'val': 250, 'test': 500}
print('Dataset:', dataset['counts'])
print('Project:', ROOT)
'''

CREDENTIALS = '''from google.colab import userdata
os.environ['CLEARML_API_ACCESS_KEY'] = userdata.get('CLEARML_ACCESS_KEY')
os.environ['CLEARML_API_SECRET_KEY'] = userdata.get('CLEARML_SECRET_KEY')
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
assert os.environ['CLEARML_API_ACCESS_KEY'] and os.environ['CLEARML_API_SECRET_KEY']
print('ClearML project: Visible Ship Segmentation Study')
'''

HELPER = '''def run_one(model, smoke=True):
    config = ROOT / 'configs' / f'{model}.json'
    command = [str(PYTHON), '-u', '-m', 'sarstudy.runner', '--config', str(config),
               '--model', model, '--dataset', 'visible']
    if smoke:
        command.append('--smoke')
    environment = os.environ.copy()
    environment['MPLBACKEND'] = 'Agg'
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=environment)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(f'{model}/visible failed: exit code {return_code}; see preceding error')
    print('Completed:', model)
'''


def save_notebook(name, cells):
    value = dict(cells=cells, metadata=dict(colab=dict(name=name, provenance=[]),
        accelerator='GPU', kernelspec=dict(name='python3', display_name='Python 3'),
        language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
    folder = ROOT / 'notebooks'; folder.mkdir(exist_ok=True)
    (folder / name).write_text(json.dumps(value, indent=1))


def training_notebook(family, models, name):
    cells = [md(f'# Visible ships — {family} models\n\nRun setup cells in order, then one model at a time. Shared seed 42, 1,000 train / 250 validation / 500 test; 500 maximum epochs, patience 20. Patience and best-checkpoint selection use validation mask AP50:95. Test is evaluated after training. SAMSAR uses ground-truth box prompts and is labelled accordingly.\n'),
        md('## 1. Mount Drive and extract the files\n'), code(MOUNT),
        md('## 2. GPU and environment\n\nUse A100 or T4. MMDetection requires Python 3.11 and cannot use the Blackwell G4 GPU with these packages. Modern models can use Python 3.11–3.13.\n'),
        code('!nvidia-smi\n'),
        code(f'''subprocess.run([sys.executable, str(ROOT / 'scripts/setup_environment.py'), '{family}'], check=True)
PYTHON = Path('/content/visible-{family}-env/bin/python')
subprocess.run([str(PYTHON), '-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'], check=True)
'''),
        md('## 3. ClearML credentials\n\nSet CLEARML_ACCESS_KEY and CLEARML_SECRET_KEY in Colab Secrets.\n'), code(CREDENTIALS),
        md('## 4. Verify the shared settings\n'), code('''settings = json.loads((ROOT / 'configs/visible.json').read_text())
print('Epoch ceiling:', settings['max_epochs'], '| Patience:', settings['patience'])
print('Seed:', settings['seed'], '| Batch:', settings['batch_size'], '| AMP:', settings['amp'])
print('Split sizes:', dataset['counts'])
assert settings['seed'] == 42 and settings['max_epochs'] == 500 and settings['patience'] == 20
'''), md('## 5. Run helper\n'), code(HELPER)]
    if 'samsar' in models:
        cells.extend([md('## SAMSAR prerequisites\n\nRun this before the SAMSAR cells. Your existing SAMSAR implementation folder is required.\n'), code('''import urllib.request
repo = Path(settings['samsar_repo'])
assert (repo / 'samsar/modeling/build_samsar.py').is_file(), f'SAMSAR implementation missing: {repo}'
checkpoint = Path(settings['sam_checkpoint'])
if not checkpoint.is_file():
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.download')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', temporary)
    assert temporary.stat().st_size > 100_000_000
    temporary.replace(checkpoint)
print('SAMSAR prerequisites ready')
''')])
    cells.append(md('## 6. Two-epoch smoke tests\n\nThese check execution, saved loss history and validation logging. Low AP after two epochs does not measure final performance. Run each model cell once.\n'))
    for model in models:
        cells.extend([md(f'### {model}\n'), code(f"run_one('{model}', smoke=True)\n")])
    cells.extend([md('## 7. Full training\n\nAfter checking the smoke output, set the following value to True. On a new session, rerun setup and the helper, then rerun the relevant full-training cell to resume. Do not change settings during a run.\n'),
        code('APPROVE_FULL_TRAINING = False\n')])
    for model in models:
        cells.extend([md(f'### {model}\n'), code(f"assert APPROVE_FULL_TRAINING, 'Set APPROVE_FULL_TRAINING = True after the smoke check'\nrun_one('{model}', smoke=False)\n")])
    cells.extend([md('## 8. Confirm completed outputs\n'), code(f'''for model in {models!r}:
    folder = ROOT / 'runs/visible' / model / settings['experiment']
    print(model, 'training:', (folder / 'training_complete.json').is_file(),
          'evaluation:', (folder / 'evaluation_complete.json').is_file(),
          'history:', (folder / 'history.csv').is_file())
''')])
    save_notebook(name, cells)


def main():
    for model in MODELS:
        config = {**BASE, 'models': [model]}
        if model == 'solov2_gala': config['learning_rate'] = .01
        if model == 'swininsseg': config['learning_rate'] = .00002
        (ROOT / 'configs' / f'{model}.json').write_text(json.dumps(config, indent=2))
    (ROOT / 'configs/visible.json').write_text(json.dumps(BASE, indent=2))
    training_notebook('modern', MODELS[:6], '01_Visible_Modern_Models.ipynb')
    training_notebook('mmdetection', MODELS[6:], '02_Visible_MMDetection_Models.ipynb')
    save_notebook('03_Visible_Results.ipynb', [md('# Visible ships — final results and visualisations\n'),
        code(MOUNT), code('''%pip -q install numpy pandas matplotlib Pillow pycocotools
sys.path.insert(0, str(ROOT))
from sarstudy.config import StudyConfig
from sarstudy.reports import aggregate
settings = json.loads((ROOT / 'configs/visible.json').read_text())
results = aggregate(StudyConfig(**settings))
display(results)
'''), md('## Saved training and validation curves\n'), code('''from IPython.display import display, Image
for model in settings['models']:
    path = ROOT / 'runs/visible' / model / settings['experiment'] / 'figures/learning_curves.png'
    if path.is_file():
        print(model)
        display(Image(filename=str(path)))
'''), md('## Same test image segmented by every completed model\n'), code('''subprocess.run([sys.executable, str(ROOT / 'scripts/visualise_visible.py')], check=True)
for path in sorted((ROOT / 'reports/qualitative').glob('*.png')):
    print(path.name)
    display(Image(filename=str(path)))
''')])
    # Explicit inventory avoids including old SAR notebooks or caches.
    package = ROOT.parent / 'Visible_Ship_Study_Package.zip'
    files = [ROOT / 'pyproject.toml', ROOT / 'README_VISIBLE.md']
    for directory, pattern in [('sarstudy', '*.py'), ('environments', '*.txt'),
                               ('notebooks', '*Visible*.ipynb')]:
        files.extend((ROOT / directory).rglob(pattern))
    files.extend(ROOT / 'configs' / f'{name}.json' for name in ['visible'] + MODELS)
    files.extend(ROOT / 'scripts' / name for name in
        ['setup_environment.py', 'prepare_visible_bundle.py', 'visualise_visible.py', 'build_visible_notebooks.py',
         'verify_visible_package.py', 'run_visible.py'])
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT.parent))
    print(package)


if __name__ == '__main__': main()
