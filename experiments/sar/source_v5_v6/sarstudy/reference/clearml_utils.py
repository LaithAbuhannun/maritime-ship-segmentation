"""Optional ClearML facade; benchmark execution never depends on tracking."""
from __future__ import annotations
import os
from typing import Any

_task: Any | None = None


def init(cfg) -> Any | None:
    global _task
    if not cfg.use_clearml:
        return None
    try:
        from clearml import Task
        _task = Task.init(project_name=cfg.clearml_project,
                          task_name=cfg.clearml_run_name,
                          reuse_last_task_id=False)
        _task.connect(cfg.as_dict(), name="benchmark")
    except Exception as error:
        print(f"[ClearML disabled] {error}")
        _task = None
    return _task


def task():
    return _task


def enabled() -> bool:
    return _task is not None


def log_params(section: str, params: dict) -> None:
    if _task is not None:
        _task.connect(params, name=section)


def log_scalar(title: str, series: str, value: float, iteration: int) -> None:
    if _task is not None:
        _task.get_logger().report_scalar(title, series, float(value), int(iteration))


def log_metrics(model_name: str, metrics: dict) -> None:
    if _task is None:
        return
    logger = _task.get_logger()
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            logger.report_single_value(f"{model_name}/{metric}", float(value))


def log_table(name: str, df) -> None:
    if _task is not None:
        _task.get_logger().report_table(name, "comparison", iteration=0,
                                        table_plot=df)


def upload_artifact(name: str, path_or_obj) -> None:
    if _task is not None:
        _task.upload_artifact(name, artifact_object=path_or_obj)


def log_text(msg: str) -> None:
    print(msg)
    if _task is not None:
        _task.get_logger().report_text(msg)


def close() -> None:
    global _task
    if _task is not None:
        _task.close()
    _task = None


def setup_credentials(access_key: str, secret_key: str,
                      api_host: str = "https://api.clear.ml",
                      web_host: str = "https://app.clear.ml",
                      files_host: str = "https://files.clear.ml") -> None:
    values = {"CLEARML_API_ACCESS_KEY": access_key,
              "CLEARML_API_SECRET_KEY": secret_key,
              "CLEARML_API_HOST": api_host, "CLEARML_WEB_HOST": web_host,
              "CLEARML_FILES_HOST": files_host}
    for key, value in values.items():
        if value:
            os.environ[key] = value

