"""Create an isolated Colab environment; never replace notebook-kernel packages."""
from pathlib import Path
import argparse
import subprocess
import sys
import venv

def ensure_environment(env):
    """Repair partially created environments without replacing kernel packages."""
    python = env / 'bin/python'
    if not python.exists():
        venv.EnvBuilder(with_pip=False).create(env)
    probe = subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True)
    if probe.returncode:
        print('Installing missing pip inside the isolated environment...', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', '--python', str(python),
                        'install', 'pip'], check=True)
    subprocess.run([str(python), '-m', 'pip', '--version'], check=True)
    return python

def main():
    p=argparse.ArgumentParser(); p.add_argument('family',choices=['modern','mmdetection','reports'])
    args=p.parse_args()
    if args.family=='mmdetection' and sys.version_info[:2]!=(3,11):
        raise SystemExit('Select a Python 3.11 Colab runtime for MMDetection. Nothing was installed.')
    if not (3,11)<=sys.version_info[:2]<=(3,13):
        raise SystemExit('Use Python 3.11–3.13; MMDetection specifically requires 3.11.')
    root=Path(__file__).resolve().parents[1]
    env=Path('/content')/f'sar-{args.family}-env'
    python=ensure_environment(env)
    def pip(*options): subprocess.run([str(python),'-m','pip',*options],check=True)
    pip('install','--upgrade','pip','setuptools','wheel')
    lock=root/'environments/locks'/f'{args.family}-py{sys.version_info.major}{sys.version_info.minor}.txt'
    if lock.exists():
        extra=[]
        if args.family=='modern': extra=['--extra-index-url','https://download.pytorch.org/whl/cu128']
        if args.family=='mmdetection':
            extra=['--extra-index-url','https://download.pytorch.org/whl/cu121','--only-binary=mmcv',
                   '-f','https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html']
        pip('install','-r',str(lock),*extra)
    elif args.family=='modern':
        pip('install','torch==2.7.1','torchvision==0.22.1','--index-url',
            'https://download.pytorch.org/whl/cu128')
        pip('install','-r',str(root/'environments/modern.txt'))
    elif args.family=='mmdetection':
        pip('install','torch==2.1.0','torchvision==0.16.0','--index-url',
            'https://download.pytorch.org/whl/cu121')
        # Fail quickly if a wheel is unavailable; do not launch a silent source build.
        pip('install','--only-binary=mmcv','-f',
            'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html',
            '-r',str(root/'environments/mmdetection.txt'))
    pip('install',str(root))
    pip('check')
    if not lock.exists():
        lock.parent.mkdir(parents=True,exist_ok=True)
        resolved=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
        temporary=lock.with_suffix('.tmp'); temporary.write_text(resolved); temporary.replace(lock)
    print('READY:',python)
    print('No Colab restart is needed: run the notebook subprocess cells.')

if __name__=='__main__': main()
