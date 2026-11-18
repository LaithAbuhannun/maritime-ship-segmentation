"""Generate plain Colab notebooks; source of truth for their setup/run cells."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def md(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text): return {'cell_type':'code','metadata':{},'source':text.splitlines(True),'outputs':[], 'execution_count':None}
MOUNT="""from pathlib import Path
from google.colab import drive
import sys, subprocess, json, os
drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/SAR_Ship_Segmentation_Study')
assert (ROOT / 'sarstudy').is_dir(), f'Upload the whole project folder first: {ROOT}'
print('Project:', ROOT)
"""
AUTH="""from google.colab import userdata
# Secrets stay in this session's environment; they are never written into the package.
os.environ['CLEARML_API_ACCESS_KEY'] = userdata.get('CLEARML_ACCESS_KEY')
os.environ['CLEARML_API_SECRET_KEY'] = userdata.get('CLEARML_SECRET_KEY')
os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml'
os.environ['CLEARML_FILES_HOST'] = 'https://files.clear.ml'
print('ClearML credentials configured')
"""
CONFIG="""CONFIG = ROOT / 'configs' / 'study.json'
settings = json.loads(CONFIG.read_text())
print('Maximum epochs:', settings['max_epochs'], '| Patience:', settings['patience'])
print('Datasets:', settings['datasets'])
print('Experiment:', settings['experiment'])
# Edit configs/study.json BEFORE the first smoke run, not halfway through training.
# Epoch ceiling 500, patience 20; validation is 20% of the selected training pool.
"""
RUN="""def run_one(model, dataset, smoke=True):
    command = [str(PYTHON), '-u', '-m', 'sarstudy.runner',
               '--config', str(CONFIG), '--model', model, '--dataset', dataset]
    if smoke:
        command.append('--smoke')
    else:
        marker = ROOT/'runs'/dataset/model/(settings['experiment']+'_smoke')/'evaluation_complete.json'
        assert marker.is_file(), f'Complete the two-epoch smoke run first: {model}/{dataset}'
    subprocess.run(command, check=True, env=os.environ.copy())
"""

def build(family,models,filename,title):
    cells=[md(f'# {title}\n\nRun top to bottom. Existing Honours_Benchmark results are untouched. '
              'Start with the two-epoch smoke checks. Long training remains locked until they succeed.\n\n'
              'Training and validation curves use separate splits. Test images never control patience or checkpoint selection.'),
           md('## 1. Mount Drive'),code(MOUNT),md('## 2. Install packages\n\nRun once per session and wait for READY. '
               'This installs packages, not training. No restart is needed. If installation fails, stop here and copy '
               'the error above the final traceback; do not run the training cells.'),
           md('Use an A100 (Python 3.11) for this notebook. Do not use Blackwell with this legacy stack.' if family=='mmdetection'
              else 'Use a CUDA GPU. This modern stack targets CUDA 12.8, including Blackwell; the preflight checks actual kernels. '
                   'For comparable FPS across all models use the same GPU type, such as A100, in both training notebooks.'),
           code(f"subprocess.run([sys.executable, str(ROOT/'scripts/setup_environment.py'), '{family}'], check=True)\nPYTHON = Path('/content/sar-{family}-env/bin/python')\n"),
           md('## 3. ClearML credentials'),code(AUTH),md('## 4. Frozen study settings'),code(CONFIG)]
    if family=='modern':
        cells.extend([md('## 5. SAMSAR prerequisites\n\nThe external SAMSAR implementation must already exist in Drive. '
                         'This downloads only the official SAM ViT-B initialization checkpoint, not old trained weights.'),
                      code("""import urllib.request
repo = Path(settings['samsar_repo'])
assert (repo/'samsar/modeling/build_samsar.py').is_file(), f'SAMSAR source not found: {repo}'
checkpoint = Path(settings['sam_checkpoint'])
if not checkpoint.is_file():
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix('.download')
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth', temporary)
    temporary.replace(checkpoint)
assert checkpoint.stat().st_size > 100_000_000, 'SAM checkpoint looks incomplete'
print('SAMSAR prerequisites ready')
""")])
    cells.extend([md('## Run helpers'),code(RUN),
                  md('## Two-epoch verification\n\nRun one cell at a time. Check that train_loss, val_loss and validation AP '
                     'appear and that ClearML has the Loss / total graph. A smoke pass checks execution, not convergence.' )])
    for model in models:
        cells.append(code(f"for dataset in settings['datasets']:\n    run_one('{model}', dataset, smoke=True)\n"))
    cells.extend([md('## Long runs — only after checking smoke results\n\nEnable this deliberately. Each model/dataset has its own '
                    'history and checkpoint folder. If Colab disconnects, repeat setup and rerun the same cell; saved state is resumed. '
                    'Do not run two copies of the same experiment simultaneously.'),
                  code("APPROVE_LONG_RUNS = False  # Change to True only after checking smoke logs and curves.\n")])
    for model in models:
        cells.append(code(f"assert APPROVE_LONG_RUNS, 'Inspect smoke results first'\nfor dataset in settings['datasets']:\n    run_one('{model}', dataset, smoke=False)\n"))
    cells.extend([md('## Check durable outputs'),code("""for dataset in settings['datasets']:
    for model in settings['models']:
        folder = ROOT/'runs'/dataset/model/settings['experiment']
        print(dataset, model, 'training:', (folder/'training_complete.json').is_file(),
              'evaluation:', (folder/'evaluation_complete.json').is_file())
""")])
    write(filename,cells)

def write(filename,cells):
    payload={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
             'colab':{'name':filename}},'nbformat':4,'nbformat_minor':5}
    for index,cell in enumerate(cells): cell['id']=f'cell-{index:03d}'
    (ROOT/'notebooks'/filename).write_text(json.dumps(payload,indent=2))

if __name__=='__main__':
    build('modern',['yolo26n_seg','yolo26s_seg','yolo26m_seg','rfdetr_seg','samsar','ship_contour'],
          '01_Modern_Models.ipynb','01 · Modern segmentation models')
    build('mmdetection',['swininsseg','solov2_gala'],'02_MMDetection_Models.ipynb','02 · MMDetection models')
    write('03_Results_and_Curves.ipynb',[
        md('# 03 · Results and learning curves\n\nCPU is sufficient. Reads individual run files; never overwrites them.'),
        code(MOUNT),code("subprocess.run([sys.executable,str(ROOT/'scripts/setup_environment.py'),'reports'],check=True)\nPYTHON=Path('/content/sar-reports-env/bin/python')\n"),
        code("""script = '''from sarstudy.config import StudyConfig
from sarstudy.storage import read_json
from sarstudy.reports import aggregate
import sys
aggregate(StudyConfig(**read_json(sys.argv[1])))
'''
subprocess.run([str(PYTHON),'-c',script,str(ROOT/'configs/study.json')],check=True)
"""),code("""import pandas as pd
settings=json.loads((ROOT/'configs/study.json').read_text())
reports=ROOT/'reports'/settings['experiment']
for name in ['segmentation.csv','box_generalization.csv']:
    print(name)
    try:
        display(pd.read_csv(reports/name))
    except pd.errors.EmptyDataError:
        print('No completed metrics for this table yet.')
print('Missing:',json.loads((reports/'missing.json').read_text()))
"""),code("""from IPython.display import Image, display
DATASET='ssdd'
MODEL='ship_contour'
curve=ROOT/'runs'/DATASET/MODEL/settings['experiment']/'figures/learning_curves.png'
if curve.is_file(): display(Image(filename=str(curve)))
else: print('This run has not exported its curves yet:',curve)
""")])
