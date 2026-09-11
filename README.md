# ML Learning Lab Runpod Worker v5

Perubahan utama:
- mempertahankan auto-resolve folder wrapper dataset classification;
- membersihkan output terminal/ANSI;
- menangkap matplotlib plots;
- mengirim `summary` eksperimen: device GPU, model, jumlah parameter, durasi, jumlah citra/kelas, dan metrik numerik dari `RESULT`/Keras History.

Upload `handler.py`, `requirements.txt`, dan `Dockerfile` ke repository worker Runpod lalu deploy/release ulang endpoint.
