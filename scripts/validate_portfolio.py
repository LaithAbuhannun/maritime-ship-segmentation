"""Validate public evidence, links and privacy before publishing; standard library only."""
from pathlib import Path
import ast,csv,json,math,re,sys
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]

class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src','data-src') and value:self.targets.append(value)

def main():
    errors=[]
    rows=json.loads((ROOT/'results/selected_test_metrics.json').read_text())
    site_rows=json.loads((ROOT/'docs/assets/metrics.json').read_text())
    if rows!=site_rows:errors.append('Website and downloadable metric JSON differ')
    if len(rows)!=22:errors.append('Expected 22 selected test metric rows')
    if len({(r['dataset'],r['model']) for r in rows})!=len(rows):errors.append('Duplicate selected model/dataset identities')
    for row in rows:
        for key in ['AP50_95','AP50','AP75','Precision','Recall','F1']:
            value=row[key]
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=100:errors.append(f'Invalid {key}: {row["dataset"]}/{row["model"]}')
        if row.get('smoke'):errors.append('A smoke result was included as final')
        if row['model']=='samsar' and row['protocol']!='ground_truth_box_prompted':errors.append('SAMSAR protocol not explicit')
    expected={'ssdd':('rfdetr_seg',62.284),'hrsid':('yolo26m_seg',38.222),'visible':('rfdetr_seg',70.755)}
    for ds,(model,score) in expected.items():
        winner=max((r for r in rows if r['dataset']==ds and r['model']!='samsar'),key=lambda r:r['AP50_95'])
        if winner['model']!=model or abs(winner['AP50_95']-score)>.001:errors.append(f'Unexpected selected winner for {ds}')
    csv_rows=list(csv.DictReader((ROOT/'results/segmentation_metrics.csv').open()))
    if len(csv_rows)!=len(rows):errors.append('CSV has a different row count')
    for j,row in enumerate(csv_rows):
        for key in ['AP50_95','AP50','Precision','Recall','F1','FPS']:
            if float(row[key])!=rows[j][key]:errors.append(f'CSV disagreement at row {j}: {key}')
    if len(list((ROOT/'results/histories').glob('*.json')))!=24:errors.append('Expected 24 retained histories')
    parser=Links();parser.feed((ROOT/'docs/index.html').read_text())
    for link in parser.targets:
        path=link.split('#')[0].split('?')[0]
        if path and not re.match(r'^[a-z]+:',path) and not (ROOT/'docs'/path).is_file():errors.append(f'Missing site asset: {path}')
    for p in ROOT.rglob('*.py'):
        try:ast.parse(p.read_text(),filename=str(p))
        except SyntaxError as exc:errors.append(f'Python syntax: {p.relative_to(ROOT)}: {exc}')
    secrets=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'sk-(?:proj-)?[A-Za-z0-9_-]{35,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',r'AKIA[0-9A-Z]{16}']
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts or p.suffix not in {'.py','.ipynb','.json','.md','.txt','.html','.js','.yml','.yaml','.toml'}:continue
        text=p.read_text()
        if ('2486'+'9570') in text:errors.append(f'Student number present: {p.relative_to(ROOT)}')
        if any(re.search(pattern,text) for pattern in secrets):errors.append(f'Potential credential: {p.relative_to(ROOT)}')
        if p.suffix=='.ipynb':
            try:json.loads(text)
            except json.JSONDecodeError:errors.append(f'Invalid notebook JSON: {p.relative_to(ROOT)}')
    if errors:
        print('\n'.join(errors));return 1
    print('PASS: 22 selected rows, CSV/site agreement, 24 histories, expected rankings, local assets, Python syntax, notebook JSON and credential scan.')
    print('This does not repeat GPU training or prove scientific reproducibility.')
    return 0

if __name__=='__main__':sys.exit(main())
