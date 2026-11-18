"""Durable epoch ledger first; explicit ClearML reporting second."""
import csv
import math
import warnings
from pathlib import Path
from .storage import atomic_json, read_json

class EarlyStop:
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs, self.best_epoch = None, 0, 0

    def update(self, metric, epoch):
        if not math.isfinite(metric):
            raise ValueError("Non-finite validation AP")
        improved = self.best is None or metric > self.best + self.min_delta
        if improved:
            self.best, self.best_epoch, self.bad_epochs = metric, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self):
        return dict(best=self.best, best_epoch=self.best_epoch, bad_epochs=self.bad_epochs)

    def load_state_dict(self, state):
        self.best, self.best_epoch, self.bad_epochs = state['best'], state['best_epoch'], state['bad_epochs']

class Tracker:
    def __init__(self, cfg, dataset, model, run_dir):
        self.folder = Path(run_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder/'history.json'
        self.rows = read_json(self.path) if self.path.exists() else []
        self.task = None
        self.pending = []
        if cfg.clearml:
            from clearml import Task
            attempts_path=self.folder/'clearml_attempts.json'
            attempts=read_json(attempts_path) if attempts_path.exists() else []
            # No silent fallback: authentication must succeed before GPU training.
            self.task = Task.init(project_name=cfg.clearml_project,
                task_name=f'{cfg.experiment} | {dataset} | {model} | attempt {len(attempts)+1:02d}',
                reuse_last_task_id=False, auto_connect_frameworks=False)
            self.task.connect(cfg.as_dict(), name='study')
            atomic_json(self.folder/'clearml_task.json', {'id':self.task.id})
            attempts.append({'id':self.task.id,'attempt':len(attempts)+1})
            atomic_json(attempts_path,attempts)

    def _report_row(self, row):
        if self.task is None:
            return
        logger = self.task.get_logger()
        for key, value in row.items():
            if key == 'epoch' or not isinstance(value, (float,int)) or not math.isfinite(value):
                continue
            if key in ('train_loss','val_loss'):
                title, series = 'Loss / total', key.split('_')[0]
            else:
                title, series = ('Validation metrics', key) if key.startswith('val_') else ('Training details', key)
            logger.report_scalar(title, series, float(value), int(row['epoch']))

    def epoch(self, row):
        for key in ('epoch','train_loss','val_loss','val_mask_AP50_95'):
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError(f"Required epoch field missing/non-finite: {key}")
        # An interrupted epoch may be replayed; replace that epoch instead of duplicating it.
        self.rows = [r for r in self.rows if r['epoch'] < row['epoch']] + [row]
        atomic_json(self.path, self.rows)
        fields = sorted({k for r in self.rows for k in r})
        temp = self.folder/'history.csv.tmp'
        with temp.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.rows)
        temp.replace(self.folder/'history.csv')
        from .reports import plot_history
        plot_history(self.folder)
        try:
            self._report_row(row)
            if self.task:
                self.task.flush(wait_for_uploads=True)
        except Exception as exc:
            self.pending.append(int(row['epoch']))
            atomic_json(self.folder/'clearml_pending.json', self.pending)
            warnings.warn(f"ClearML sync failed; Drive history is saved: {exc}")
        print(f"Epoch {row['epoch']}: train_loss={row['train_loss']:.5f} "
              f"val_loss={row['val_loss']:.5f} val_mask_AP={row['val_mask_AP50_95']:.3f}", flush=True)

    def metrics(self, split, values):
        atomic_json(self.folder/'metrics'/f'{split}.json', values)
        if self.task:
            for k,v in values.items():
                if isinstance(v,(int,float)) and math.isfinite(v):
                    self.task.get_logger().report_single_value(f'{split}/{k}',v)
            self.task.upload_artifact(f'{split}_metrics', artifact_object=str(self.folder/'metrics'/f'{split}.json'))

    def artifact(self, name, path):
        if self.task:
            self.task.upload_artifact(name, artifact_object=str(path))

    def close(self):
        if self.task:
            try:
                # Replay durable history also restores logging following a disconnected session.
                for row in self.rows:
                    self._report_row(row)
                for filename in ('history.json','history.csv','config.json','identity.json','clearml_attempts.json'):
                    path = self.folder/filename
                    if path.exists(): self.artifact(filename, path)
                self.task.flush(wait_for_uploads=True)
                atomic_json(self.folder/'clearml_pending.json', [])
            finally:
                self.task.close()
