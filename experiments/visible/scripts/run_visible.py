"""CLI entry point: python scripts/run_visible.py MODEL [--smoke]."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['yolo26n_seg', 'yolo26s_seg', 'yolo26m_seg',
        'rfdetr_seg', 'samsar', 'ship_contour', 'swininsseg', 'solov2_gala'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-u', '-m', 'sarstudy.runner', '--dataset', 'visible',
               '--model', args.model, '--config', str(ROOT / 'configs' / f'{args.model}.json')]
    if args.smoke: command.append('--smoke')
    raise SystemExit(subprocess.call(command))


if __name__ == '__main__': main()
