
# Klasifikasi SMS Spam dan Non-Spam Menggunakan TF-IDF dan MLP

Proyek kelompok mata kuliah Pengantar Algoritma Deep Learning — Tahap UTS.

## Deskripsi
Sistem klasifikasi biner SMS berbahasa Inggris untuk membedakan pesan spam dan non-spam (ham), menggunakan TF-IDF sebagai representasi teks dan Multilayer Perceptron (MLP) sebagai model dasar Deep Learning.

## Dataset
- Sumber: UCI SMS Spam Collection
- URL: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Lisensi: CC BY 4.0
- Data awal: 5.574 pesan
- Duplikat dihapus: 415
- Data akhir: 5.159 pesan unik
- Ham: 4.517; Spam: 642

Pembagian data menggunakan stratified split, seed 42:
- Training: 4.127
- Validation: 516
- Test: 516 (belum dievaluasi)

Dataset mentah dan hasil split tidak disertakan dalam repositori.

## Metode
- TF-IDF: maksimal 2.000 fitur, n-gram (1, 2)
- Model: Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(1, Sigmoid)
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Maximum epochs: 15, dengan early stopping

## Instalasi dan Menjalankan
Persyaratan: Python 3.13, Windows/lingkungan Python yang kompatibel.

1. Unduh dataset dari situs UCI.
2. Letakkan file `SMSSpamCollection` di `data/raw/`.
3. Buka terminal pada direktori utama proyek.
4. Jalankan perintah berikut secara berurutan.

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-hari1.txt
    .\.venv\Scripts\python.exe -m pip install -r requirements-tahap2.txt
    .\.venv\Scripts\python.exe scripts\02_audit_dataset_en.py
    .\.venv\Scripts\python.exe scripts\03_split_tfidf.py
    .\.venv\Scripts\python.exe scripts\04_train_mlp.py
    .\.venv\Scripts\python.exe scripts\05_analyze_validation_errors.py

## Hasil Evaluasi Validation
- Accuracy: 99,03%
- Precision spam: 95,38%
- Recall spam: 96,88%
- F1-score spam: 96,12%
- Confusion matrix: [[449, 3], [2, 62]]

Seluruh metrik berasal dari validation set, bukan test set.
Sebagai pembanding sederhana, prediksi selalu-ham menghasilkan accuracy 87,60% dan F1-score spam 0%.

## Grafik Hasil
Grafik distribusi dataset, proses pelatihan, dan confusion matrix tersedia pada `outputs/figures/`.

## Pengembangan UAS
Rencana pengembangan: model LSTM/GRU, hyperparameter tuning, studi ablasi, evaluasi lanjutan, dan aplikasi klasifikasi menggunakan Gradio.

## Catatan
Dataset tidak boleh didistribusikan ulang tanpa memperhatikan lisensinya. File yang mengandung teks SMS atau informasi pribadi tidak dipublikasikan dalam repositori.
