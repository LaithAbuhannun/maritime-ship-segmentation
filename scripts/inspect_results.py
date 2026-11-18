"""Print retained test metrics; no GPU or third-party dependency required."""
from pathlib import Path
import argparse,json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['ssdd','hrsid','visible'],default='ssdd')
    parser.add_argument('--include-prompted',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    rows=json.loads((root/'results/selected_test_metrics.json').read_text())
    rows=[r for r in rows if r['dataset']==args.dataset and (args.include_prompted or r['model']!='samsar')]
    rows.sort(key=lambda r:(r['model']=='samsar',-r['AP50_95']))
    print(f'{args.dataset.upper()} — held-out test, accuracy in %')
    print(f'{"Model":20} {"AP50:95":>9} {"Precision":>10} {"Recall":>9} {"F1":>9} {"FPS":>9}')
    for row in rows:
        print(f'{row["model"]:20} '+''.join(f'{row[k]:10.2f}' for k in ['AP50_95','Precision','Recall','F1','FPS']))
    print('\nSAMSAR receives ground-truth boxes. Some SAR FPS results used different GPUs.')
    print('SAR SOLOv2 diagnostic zero-AP runs are retained as histories, not ranked here.')

if __name__=='__main__':main()
