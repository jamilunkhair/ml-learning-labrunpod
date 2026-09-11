"""Runpod Serverless worker for ML Learning Lab.

Runpod's GitHub deploy scanner looks for the canonical entrypoint below:
    runpod.serverless.start({"handler": handler})
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path

import requests
import runpod


def _safe_extract(zip_path: str, destination: str) -> None:
    """Extract ZIP while preventing path traversal."""
    dest = Path(destination).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(dest)


def _prepare_dataset(url: str) -> tuple[str, str]:
    """Download presigned S3 ZIP and return (dataset_path, temp_root)."""
    root = tempfile.mkdtemp(prefix="ml_learning_lab_")
    zip_path = os.path.join(root, "dataset.zip")
    dataset_path = os.path.join(root, "dataset")
    os.makedirs(dataset_path, exist_ok=True)

    with requests.get(url, stream=True, timeout=(20, 300)) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)

    _safe_extract(zip_path, dataset_path)
    return dataset_path, root


def _json_safe(value):
    """Best-effort conversion for values returned by student code."""
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def handler(job):
    """Run one ML Learning Lab final-project job."""
    payload = job.get("input") or {}
    code = (payload.get("code") or "").strip()
    if not code:
        return {"ok": False, "error": "Kode Final Project kosong."}

    stdout = io.StringIO()
    stderr = io.StringIO()
    temp_root = None

    try:
        dataset_path = ""
        dataset_url = (payload.get("dataset_url") or "").strip()
        if dataset_url:
            dataset_path, temp_root = _prepare_dataset(dataset_url)

        task = payload.get("task") or "classification"
        os.environ["DATASET_PATH"] = dataset_path
        os.environ["ML_TASK"] = str(task)
        os.environ.setdefault("MPLBACKEND", "Agg")

        scope = {
            "__name__": "__main__",
            "DATASET_PATH": dataset_path,
            "ML_TASK": task,
        }

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "student_final_project.py", "exec"), scope, scope)

        # Optional convention: student code can assign RESULT = {...}
        result = _json_safe(scope.get("RESULT")) if "RESULT" in scope else None
        return {
            "ok": True,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "result": result,
        }
    except Exception:
        return {
            "ok": False,
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "error": traceback.format_exc(),
        }
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


# IMPORTANT: keep this canonical Runpod entrypoint literal in the repository.
# It is intentionally inside __main__ so importing this module in tests does
# not start a worker, while `python handler.py` (Docker CMD) does.
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
