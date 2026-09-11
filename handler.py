"""Runpod Serverless worker for ML Learning Lab v3.

Adds clean terminal output and captures matplotlib figures so remote GPU results
can be displayed inside the ML Learning Lab UI, closer to a notebook experience.
"""
import base64
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path

import requests
import runpod

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _visible_dirs(path: Path):
    try:
        return [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "__MACOSX"]
    except Exception:
        return []


def _direct_images(path: Path):
    try:
        return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    except Exception:
        return []


def _has_images_recursive(path: Path) -> bool:
    try:
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                return True
    except Exception:
        pass
    return False


def _resolve_classification_root(dataset_path: str) -> str:
    """Remove common single wrapper folders from extracted classification ZIPs.

    Example: extracted/dataset_name/cat/*.jpg + dog/*.jpg should expose
    extracted/dataset_name as DATASET_PATH, not extracted (which Keras would
    interpret as a single class named dataset_name).
    """
    current = Path(dataset_path).resolve()
    for _ in range(8):
        dirs = _visible_dirs(current)
        direct_imgs = _direct_images(current)
        # If this level already has 2+ plausible class directories, keep it.
        class_dirs = [d for d in dirs if _has_images_recursive(d)]
        if len(class_dirs) >= 2:
            return str(current)
        # Common ZIP wrapper: one directory, no images at this level.
        if not direct_imgs and len(dirs) == 1 and _has_images_recursive(dirs[0]):
            current = dirs[0]
            continue
        return str(current)
    return str(current)


def _clean_console(value: str) -> str:
    text = ANSI_RE.sub("", str(value or ""))
    out = []
    for line in text.split("\n"):
        # Keep only the latest redraw of progress lines (\r).
        out.append(line.split("\r")[-1])
    text = "\n".join(out)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    return text.strip()


def _safe_extract(zip_path: str, destination: str) -> None:
    dest = Path(destination).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(dest)


def _prepare_dataset(url: str) -> tuple[str, str]:
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
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def _capture_matplotlib_plots(max_plots: int = 8):
    plots = []
    try:
        import matplotlib.pyplot as plt
        for number in list(plt.get_fignums())[:max_plots]:
            fig = plt.figure(number)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plots.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
            buf.close()
        plt.close("all")
    except Exception:
        pass
    return plots


def handler(job):
    payload = job.get("input") or {}
    code = (payload.get("code") or "").strip()
    if not code:
        return {"ok": False, "error": "Kode Final Project kosong."}

    stdout = io.StringIO()
    stderr = io.StringIO()
    temp_root = None
    try:
        dataset_path = ""
        task = payload.get("task") or "classification"
        dataset_url = (payload.get("dataset_url") or "").strip()
        if dataset_url:
            dataset_path, temp_root = _prepare_dataset(dataset_url)
            if str(task).lower() == "classification":
                dataset_path = _resolve_classification_root(dataset_path)

        os.environ["DATASET_PATH"] = dataset_path
        os.environ["ML_TASK"] = str(task)
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        scope = {"__name__": "__main__", "DATASET_PATH": dataset_path, "ML_TASK": task}

        # In a headless worker, keep figures alive so the UI can receive them.
        try:
            import matplotlib.pyplot as plt
            plt.show = lambda *args, **kwargs: None
        except Exception:
            pass

        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(code, "student_final_project.py", "exec"), scope, scope)

        result = _json_safe(scope.get("RESULT")) if "RESULT" in scope else None
        plots = _capture_matplotlib_plots()
        return {
            "ok": True,
            "stdout": _clean_console(stdout.getvalue())[-50000:],
            "stderr": _clean_console(stderr.getvalue())[-12000:],
            "result": result,
            "plots": plots,
        }
    except Exception:
        plots = _capture_matplotlib_plots()
        return {
            "ok": False,
            "stdout": _clean_console(stdout.getvalue())[-50000:],
            "stderr": _clean_console(stderr.getvalue())[-12000:],
            "plots": plots,
            "error": traceback.format_exc(),
        }
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
