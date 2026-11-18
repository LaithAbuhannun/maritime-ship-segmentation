"""Read per-run metric files without ever overwriting model results."""
from pathlib import Path
from .storage import read_json, atomic_json

def plot_history(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder); rows=read_json(folder/'history.json')
    epochs=[r['epoch'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key,label in [('train_loss','Training'),('val_loss','Validation')]:
        axes[0].plot(epochs,[r[key] for r in rows],label=label)
    axes[0].set(xlabel='Epoch',ylabel='Total loss',title='Training and validation loss')
    axes[0].legend()
    axes[1].plot(epochs,[r['val_mask_AP50_95'] for r in rows])
    axes[1].set(xlabel='Epoch',ylabel='Mask AP50:95 (%)',title='Validation only')
    for axis in axes: axis.grid(alpha=.2)
    fig.tight_layout(); output=folder/'figures'; output.mkdir(exist_ok=True)
    paths=[output/'learning_curves.png',output/'learning_curves.pdf']
    for path in paths: fig.savefig(path,dpi=180)
    plt.close(fig)
    return paths

def aggregate(cfg):
    import pandas as pd
    root=Path(cfg.project_root); output=root/'reports'/cfg.experiment
    output.mkdir(parents=True,exist_ok=True)
    rows,missing=[],[]
    for dataset in cfg.datasets:
        for model in cfg.models:
            folder=cfg.run_dir(dataset,model)
            for split in ['test']+(['ls_ssdd'] if dataset=='ssdd' and model!='samsar' else []):
                path=folder/'metrics'/f'{split}.json'
                if not path.exists():
                    missing.append({'dataset':dataset,'model':model,'split':split}); continue
                result=read_json(path)
                if result['model']!=model or result['training_dataset']!=dataset:
                    raise ValueError(f'Wrong result identity: {path}')
                rows.append({**result,'source_file':str(path)})
    atomic_json(output/'all_results.json',rows)
    atomic_json(output/'missing.json',missing)
    atomic_json(output/'fps_comparability.json',{
        'gpu_types':sorted({r.get('gpu','unknown') for r in rows}),
        'direct_hardware_comparison':len({r.get('gpu','unknown') for r in rows})==1 and bool(rows),
        'note':'Compare FPS only on identical hardware and timing protocol; prompted SAMSAR is separate.'})
    scalar=[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in rows]
    frame=pd.DataFrame(scalar)
    for split,field in [('segmentation','AP50_95'),('box_generalization','box_F1')]:
        selected=frame[frame[field].notna()] if field in frame else pd.DataFrame()
        selected.to_csv(output/f'{split}.csv',index=False)
    print('Missing:',missing or 'None')
    print('Reports:',output)
    return frame
