# ML Learning Lab - Runpod Serverless Worker

Upload these files to the **root** of the GitHub repository:

- `handler.py`
- `requirements.txt`
- `Dockerfile`

Runpod GitHub deploy settings:

- Branch: `main`
- Dockerfile path: `/Dockerfile`
- Endpoint type: `Queue`

The canonical Runpod Serverless entrypoint is present literally in `handler.py`:

```python
runpod.serverless.start({"handler": handler})
```
