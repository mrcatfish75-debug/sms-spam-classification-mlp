# SMS Spam UTS — Tahap 2: Split, TF-IDF, Baseline MLP

**Proyek:** `D:\1. PROJEk DL\SMS-Spam-Indonesia`
**Input terverifikasi pengguna:** `data/processed/sms_en_clean.csv`, 5.159 baris dengan kolom `source_row,label,text,panjang_karakter,panjang_kata`.
**Bahasa:** Inggris asli, UCI SMS Spam Collection. **Jalur:** B (topik mandiri).

## Cara instalasi di PowerShell

1. Ekstrak ZIP **ke root folder proyek**. Ini hanya menambah file baru: `scripts/03_split_tfidf.py`, `scripts/04_train_mlp.py`, `requirements-tahap2.txt`, `PANDUAN_TAHAP2_UTS.md`.
2. Buka VS Code terminal, jalankan:

```powershell
Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
Test-Path 'data\processed\sms_en_clean.csv'
Test-Path 'scripts\03_split_tfidf.py'
Test-Path 'scripts\04_train_mlp.py'
.\.venv\Scripts\python.exe -m pip install -r requirements-tahap2.txt
.\.venv\Scripts\python.exe -c "import sklearn, tensorflow as tf; print('SKLEARN:', sklearn.__version__, 'TF:', tf.__version__)"
```

TensorFlow berukuran besar (Windows CPU) dan dapat memasang versi NumPy yang kompatibel; jangan batalkan instalasi selama tidak ada error.

## Tahap 2A — Split dan TF-IDF

```powershell
.\.venv\Scripts\python.exe scripts\03_split_tfidf.py
Get-Content 'outputs\reports\split_tfidf_manifest.json' -Encoding UTF8
```

- Data dipecah **80:10:10** dengan stratifikasi menurut kelas, seed 42.
- Deduplikasi tambahan case/whitespace diperiksa sebelum split, konflik label akan menghentikan proses untuk review.
- `TfidfVectorizer` **fit hanya pada TRAIN**, validasi di-transform dengan vectorizer yang sama.
- Test dikunci untuk evaluasi final UAS; jangan gunakan untuk mencoba-coba parameter.
- Mencatat SHA256 sumber, ukuran partisi, distribusi kelas, dan cek tidak ada overlap.

## Tahap 2B — Training MLP dan validasi

```powershell
.\.venv\Scripts\python.exe scripts\04_train_mlp.py
Get-Content 'outputs\reports\hasil_validasi_mlp.json' -Encoding UTF8
```

Arsitektur: `TF-IDF (maks 2.000 fitur) → Dense(64, ReLU) → Dropout(0.20) → Dense(32, ReLU) → Dense(1, Sigmoid)`.
Loss: binary crossentropy; optimizer Adam dengan lr=0,001; batch size 32; maksimal 15 epoch, early stopping patience 3. Threshold 0,5. Ini **baseline usulan**, belum pernah diuji pada dataset pengguna sebelum dieksekusi di laptop.

## Keluaran utama

```text
data/splits/sms_train.csv
data/splits/sms_validation.csv
data/splits/sms_test.csv
models/tfidf_vectorizer.joblib
models/mlp_baseline.keras
outputs/reports/split_tfidf_manifest.json
outputs/reports/hasil_validasi_mlp.json
outputs/reports/classification_report_validation.txt
outputs/reports/mlp_training_history.csv
outputs/figures/distribusi_split.png
outputs/figures/kurva_training_mlp.png
outputs/figures/confusion_matrix_validation.png
```

**Tidak menghapus berkas tahap sebelumnya.** Jangan unggah dataset mentah/bersih atau data pribadi ke repositori publik sebelum aturan distribusi dan isi data diperiksa. Jangan klaim hasil TEST sebelum evaluasi akhir. Catat persetujuan dosen topik mandiri.

## Pemeriksaan hasil untuk dikirim

Kirim dua file: `outputs/reports/split_tfidf_manifest.json` dan `outputs/reports/hasil_validasi_mlp.json` (atau salinan output terminal jika ada error). Tidak perlu mengirim model `.keras` jika belum diminta.

## Catatan penting

- Baseline UTS dibandingkan secara deskriptif dengan prediksi mayoritas `ham`, bukan dengan skor yang dikarang.
- TF-IDF word unigram+bigram, satu-karakter token dipertahankan untuk SMS informal.
- Confusion matrix yang dibuat adalah **validation**, belum **test**.
- Model tidak diklaim mampu memprediksi SMS Indonesia ataupun pesan modern tanpa pengujian terpisah.
- Training pada CPU dan RAM 8 GB layak dicoba dengan konfigurasi ini, tetapi performa/waktu belum dibenchmark pada laptop pengguna.
