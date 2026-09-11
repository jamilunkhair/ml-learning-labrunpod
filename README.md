# ML Learning Lab Runpod Worker v4

Perbaikan dataset classification:
- otomatis melewati folder pembungkus tunggal hasil ekstraksi ZIP;
- `DATASET_PATH` diarahkan ke folder yang benar-benar berisi folder kelas;
- mencegah Keras membaca nama folder ZIP sebagai satu-satunya kelas;
- tetap membersihkan output terminal dan mengirim plot matplotlib.

Setelah mengganti file worker di GitHub, build/release ulang endpoint Runpod.
