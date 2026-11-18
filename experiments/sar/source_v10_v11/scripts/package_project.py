"""Build a source-only upload archive; never include data, weights, tokens or caches."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'sarstudy','scripts','environments','configs','docs','notebooks','tests'}

def package():
    destination=ROOT.parent/(ROOT.name+'.zip')
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file(): continue
            relative=path.relative_to(ROOT)
            if any(x.startswith('.') or x in {'__pycache__','locks'} for x in relative.parts): continue
            if path.suffix in {'.pyc','.pt','.pth','.ckpt'}: continue
            if len(relative.parts)>1 and relative.parts[0] not in ALLOWED:
                if path.name!='README.md': continue
            if len(relative.parts)==1 and path.name not in {'README.md','pyproject.toml'}: continue
            archive.write(path,str(Path(ROOT.name)/relative))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        notebooks=[n for n in archive.namelist() if n.endswith('.ipynb')]
        assert len(notebooks)==3
    print(destination)
    print('Bytes:',destination.stat().st_size)

if __name__=='__main__': package()
