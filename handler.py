import os, io, zipfile, tempfile, traceback, contextlib, requests
import runpod


def _prepare_dataset(url):
    root=tempfile.mkdtemp(prefix='ml_lab_')
    zpath=os.path.join(root,'dataset.zip')
    r=requests.get(url,timeout=120)
    r.raise_for_status()
    open(zpath,'wb').write(r.content)
    data=os.path.join(root,'dataset')
    os.makedirs(data,exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data)
    return data


def handler(job):
    inp=job.get('input') or {}
    code=(inp.get('code') or '').strip()
    if not code:
        return {'ok':False,'error':'code kosong'}
    out,err=io.StringIO(),io.StringIO()
    try:
        dataset_path=_prepare_dataset(inp['dataset_url']) if inp.get('dataset_url') else ''
        os.environ['DATASET_PATH']=dataset_path
        os.environ['ML_TASK']=inp.get('task','classification')
        scope={'DATASET_PATH':dataset_path,'ML_TASK':os.environ['ML_TASK']}
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code,'student_final_project.py','exec'),scope,scope)
        return {'ok':True,'stdout':out.getvalue(),'stderr':err.getvalue()}
    except Exception:
        return {'ok':False,'stdout':out.getvalue(),'stderr':err.getvalue(),'error':traceback.format_exc()}

runpod.serverless.start({'handler':handler})
