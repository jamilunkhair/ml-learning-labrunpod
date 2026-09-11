"""Runpod Serverless worker for ML Learning Lab v6.

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
import time
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



def _dataset_summary(dataset_path: str):
    root=Path(dataset_path) if dataset_path else None
    if not root or not root.exists(): return 0,0
    images=[]
    try: images=[p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
    except Exception: pass
    classes=set()
    for p in images:
        try:
            rel=p.relative_to(root)
            if len(rel.parts)>=2: classes.add(rel.parts[0])
        except Exception: pass
    return len(images),len(classes)


def _detect_gpu():
    try:
        import torch
        if torch.cuda.is_available(): return torch.cuda.get_device_name(0)
    except Exception: pass
    try:
        import tensorflow as tf
        g=tf.config.list_physical_devices("GPU")
        if g: return g[0].name.replace("/physical_device:","")
    except Exception: pass
    return "CPU / GPU tidak terdeteksi"


def _numeric_metrics_from_result(result):
    out={}
    if isinstance(result,dict):
        candidates=result.get("metrics") if isinstance(result.get("metrics"),dict) else result
        for k,v in candidates.items():
            if isinstance(v,(int,float)) and not isinstance(v,bool): out[str(k)]=float(v)
    return out




def _generate_standard_classification_plots(scope):
    """Generate standard Step-10 evaluation figures when model/history/val_ds exist."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import label_binarize

        model=scope.get("model")
        val_ds=scope.get("val_ds") or scope.get("validation_ds") or scope.get("test_ds")
        hist=scope.get("history") or scope.get("hist")
        class_names=scope.get("CLASS_NAMES") or scope.get("class_names")
        if model is None or val_ds is None:
            return {}

        h=getattr(hist,"history",{}) if hist is not None else {}
        if isinstance(h,dict) and h:
            if h.get("accuracy") or h.get("val_accuracy"):
                plt.figure(figsize=(10,5.4))
                if h.get("accuracy"): plt.plot(h["accuracy"],marker="o",label="Train Accuracy")
                if h.get("val_accuracy"): plt.plot(h["val_accuracy"],marker="o",label="Validation Accuracy")
                plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy dan Validation Accuracy")
                plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
            if h.get("loss") or h.get("val_loss"):
                plt.figure(figsize=(10,5.4))
                if h.get("loss"): plt.plot(h["loss"],marker="o",label="Train Loss")
                if h.get("val_loss"): plt.plot(h["val_loss"],marker="o",label="Validation Loss")
                plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss dan Validation Loss")
                plt.grid(alpha=.25); plt.legend(); plt.tight_layout()

        ys=[]
        for _x,y in val_ds:
            try: ys.append(np.asarray(y))
            except Exception: pass
        if not ys: return {}
        y_true=np.concatenate(ys,axis=0).astype(int).reshape(-1)
        y_prob=np.asarray(model.predict(val_ds,verbose=0))
        if y_prob.ndim==1:
            y_prob=np.column_stack([1-y_prob,y_prob])
        if y_prob.ndim==2 and y_prob.shape[1]==1:
            p=y_prob[:,0]; y_prob=np.column_stack([1-p,p])
        y_pred=np.argmax(y_prob,axis=1)
        nclasses=int(y_prob.shape[1])
        if not class_names or len(class_names)!=nclasses:
            class_names=[str(i) for i in range(nclasses)]

        metrics={
            "Accuracy": float(accuracy_score(y_true,y_pred)),
            "Precision": float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
            "Recall": float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
            "F1-score": float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        }

        cm=confusion_matrix(y_true,y_pred,labels=list(range(nclasses)))
        plt.figure(figsize=(7.2,6.2))
        plt.imshow(cm,cmap="Blues",aspect="equal")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.xticks(range(nclasses),class_names,rotation=45,ha="right")
        plt.yticks(range(nclasses),class_names)
        threshold=(cm.max()/2.0) if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j,i,int(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>threshold else "black")
        plt.tight_layout()

        plt.figure(figsize=(8.5,6.2))
        auc_values=[]
        if nclasses==2:
            fpr,tpr,_=roc_curve(y_true,y_prob[:,1]); a=auc(fpr,tpr); auc_values.append(a)
            plt.plot(fpr,tpr,label=f"ROC AUC = {a:.3f}")
        else:
            y_bin=label_binarize(y_true,classes=np.arange(nclasses))
            for i,name in enumerate(class_names):
                fpr,tpr,_=roc_curve(y_bin[:,i],y_prob[:,i]); a=auc(fpr,tpr); auc_values.append(a)
                plt.plot(fpr,tpr,label=f"{name} (AUC={a:.3f})")
        plt.plot([0,1],[0,1],linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
        plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
        if auc_values: metrics["ROC-AUC"] = float(np.mean(auc_values))
        return metrics
    except Exception:
        return {}

def _runtime_summary(scope, result, dataset_path, task, elapsed):
    model=scope.get("model")
    model_name=None; params=None
    if model is not None:
        model_name=getattr(model,"name",None) or model.__class__.__name__
        try: params=int(model.count_params())
        except Exception: pass
    if not model_name and isinstance(result,dict): model_name=result.get("model") or result.get("model_name") or result.get("title")
    hist=scope.get("history") or scope.get("hist")
    metrics=_numeric_metrics_from_result(result)
    try:
        h=getattr(hist,"history",None)
        if isinstance(h,dict):
            for key,vals in h.items():
                if vals and isinstance(vals[-1],(int,float)): metrics[f"Final {key}"]=float(vals[-1])
    except Exception: pass
    images,classes=_dataset_summary(dataset_path)
    return {"task":str(task),"model":model_name or "Python Final Project","parameters":params,"gpu":_detect_gpu(),"duration_sec":round(float(elapsed),2),"dataset_images":images,"dataset_classes":classes,"metrics":metrics}

def handler(job):
    payload = job.get("input") or {}
    code = (payload.get("code") or "").strip()
    if not code:
        return {"ok": False, "error": "Kode Final Project kosong."}

    stdout = io.StringIO()
    stderr = io.StringIO()
    temp_root = None
    started=time.perf_counter()
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
        auto_metrics = _generate_standard_classification_plots(scope) if str(task).lower()=="classification" else {}
        plots = _capture_matplotlib_plots()
        summary=_runtime_summary(scope,result,dataset_path,task,time.perf_counter()-started)
        if auto_metrics:
            summary.setdefault("metrics",{}).update(auto_metrics)
        return {
            "ok": True,
            "stdout": _clean_console(stdout.getvalue())[-50000:],
            "stderr": _clean_console(stderr.getvalue())[-12000:],
            "result": result,
            "summary": summary,
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
