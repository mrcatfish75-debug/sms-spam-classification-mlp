# Tahap 3A — Analisis Kesalahan Validasi MLP UTS

## Tujuan
Mengidentifikasi lima pesan SMS yang salah diprediksi dalam evaluasi VALIDATION agar laporan dan presentasi UTS dapat membahas contoh nyata. Script **tidak membaca test set**, tidak mengubah split, tidak melatih model baru, dan tidak mengubah artefak model maupun vectorizer.

## Instalasi
Tidak ada package tambahan. Gunakan `.venv` yang telah berhasil memasang scikit-learn, joblib, pandas, numpy, dan TensorFlow.

## Petunjuk Windows PowerShell
1. Ekstrak `scripts/05_analyze_validation_errors.py` dari ZIP ke folder `scripts` dalam proyek.
2. Buka terminal VS Code dan masuk ke proyek:

```powershell
Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
```

3. Jalankan:

```powershell
.\.venv\Scripts\python.exe scripts\05_analyze_validation_errors.py
```

4. Buka output:

```powershell
code 'outputs\reports\validation_error_cases.csv'
Get-Content 'outputs\reports\validation_error_summary.json' -Encoding UTF8
```

## Output
- `outputs/reports/validation_error_cases.csv`: lima SMS yang salah diprediksi, kelas aktual, kelas prediksi, probabilitas spam, kategori kesalahan, dan kolom review manual.
- `outputs/reports/validation_error_summary.json`: ringkasan confusion matrix dan hasil pengecekan konsistensi dengan laporan baseline.

## Catatan metodologi
- False positive: SMS normal diprediksi spam. False negative: SMS spam diprediksi normal.
- Jangan memperbaiki label berdasarkan prediksi model tanpa membaca konteks SMS.
- Bila setelah review terdapat salah label atau anomali serius, dokumentasikan dan evaluasi perubahan protokol secara eksplisit. Jangan menyunting validation set diam-diam.
- Jangan membuka, mengevaluasi, atau melakukan tuning dengan test set untuk UTS ini.
- SMS asli bisa berisi nomor telepon; samarkan nomor/URL personal ketika memasukkan contoh ke laporan, slide, atau GitHub publik.
- Script memeriksa bahwa confusion matrix yang dihitung ulang **identik** dengan laporan baseline. Jika berbeda, proses berhenti tanpa menyimpan hasil. Periksa konsistensi model dan artefak sebelum meneruskan.
- Untuk slide UTS gunakan ringkasan validation: n=516, TN=449, FP=3, FN=2, TP=62, accuracy=0.990310, precision spam=0.953846, recall spam=0.968750, F1 spam=0.961240. Semua angka harus tetap disebut sebagai **VALIDATION**, bukan TEST.
