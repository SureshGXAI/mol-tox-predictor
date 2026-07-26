"""
Thin MLflow wrapper. If mlflow isn't installed (or tracking is disabled via
MLFLOW_DISABLE=1), every call becomes a no-op so training/CI still runs.

Start the UI with:  mlflow ui --backend-store-uri ./mlruns
"""
import os
from contextlib import contextmanager

_ENABLED = os.getenv("MLFLOW_DISABLE", "0") != "1"
try:
    import mlflow  # type: ignore
except Exception:
    mlflow = None
    _ENABLED = False


def enabled() -> bool:
    return _ENABLED and mlflow is not None


@contextmanager
def run(experiment: str, run_name: str = None, tracking_uri: str = None):
    if not enabled():
        yield None
        return
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as r:
        yield r


def log_params(params: dict):
    if enabled():
        mlflow.log_params(params)


def log_metrics(metrics: dict, step: int = None):
    if enabled():
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                            if v is not None}, step=step)


def log_metric(key: str, value, step: int = None):
    if enabled() and value is not None:
        mlflow.log_metric(key, float(value), step=step)


def log_artifact(path: str, artifact_path: str = None):
    if enabled() and path and os.path.exists(path):
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_dict(d: dict, filename: str):
    if enabled():
        mlflow.log_dict(d, filename)


def log_pytorch_model(model, name: str = "model"):
    if enabled():
        try:
            import mlflow.pytorch  # type: ignore
            mlflow.pytorch.log_model(model, name)
        except Exception:
            pass
