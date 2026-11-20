# UAS Tahap 2 — Training Word2Vec + GRU + Attention

Paket ini adalah lanjutan UAS Tahap 1 dan tidak mengubah dataset, hasil baseline
UTS, maupun kode pemisahan train/validation/test.

## Prasyarat yang sudah dipenuhi

- Branch lokal `uas-development`, commit awal Tahap 1 `1c9bead`.
- `data/processed/sequences_train.npz` : 4.127 × 60.
- `data/processed/sequences_validation.npz` : 516 × 60.
- `models/word2vec_embedding_matrix.npy` : 3.443 × 64.
- `models/sequence_vocabulary.json` dan `models/sequence_tokenizer_config.json`.
- Manifest `outputs/reports/uas_sequence_word2vec_manifest.json` bertanda PASS.

Script membaca bentuk dan jumlah data langsung dari manifest; tidak menebak
jumlah fitur. Jika hasil Tahap 1 berbeda, script akan memeriksa konsistensinya
sebelum melatih model. Test split sama sekali tidak dibaca.

## Cara memasang pada Windows PowerShell 5.1

1. Masuk ke `D:\1. PROJEk DL\SMS-Spam-Indonesia`.
2. Ekstrak ZIP paket Tahap 2 ke direktori proyek utama; file ada pada `scripts/07_train_gru.py`.
3. Periksa `git branch --show-current` masih `uas-development`.
4. Periksa berkas:

```powershell
Test-Path 'scripts\07_train_gru.py'
Test-Path 'requirements-uas-tahap2.txt'
Test-Path 'models\word2vec_embedding_matrix.npy'
Test-Path 'data\processed\sequences_train.npz'
Test-Path 'data\processed\sequences_validation.npz'
```

5. Semua dependency lama sudah seharusnya terinstal. Jika ada yang kurang:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap2.txt
```

6. Jalankan pelatihan (CPU):

```powershell
.\.venv\Scripts\python.exe scripts\07_train_gru.py
```

7. Periksa laporan:

```powershell
$m = Get-Content 'outputs\reports\hasil_validasi_gru_attention.json' -Raw | ConvertFrom-Json
$m.status
$m.validation_gru_attention | Format-List
$m.confusion_matrix_validation_labels_0ham_1spam | ForEach-Object { $_ -join ', ' }
$m.attention_probe | Format-List
$m.test_status
git status --short
```

## Konfigurasi awal (bukan tuning)

- Word2Vec train-only: 3.443 token × 64 dimensi, embedding dapat dilatih ulang.
- GRU: 32 unit, 1 layer, return_sequences dan final state.
- Attention: Keras AdditiveAttention dengan final state sebagai query.
- Dense 16 ReLU → Dropout 0,20 → Dense 1 Sigmoid.
- Optimizer Adam 0,001; binary cross-entropy; batch size 32.
- Maksimum 15 epoch; early stopping `val_loss`, patience 3, pulihkan bobot terbaik.
- Ambang keputusan 0,50; evaluasi validation saja.

Ini adalah konfigurasi **eksperimen awal UAS**. Hasil bukan final, dan model
belum dituning atau diuji pada test set. Tidak ada jaminan hasil mengalahkan MLP.

## Artefak lokal yang dihasilkan

- `models/gru_attention_baseline.keras` — klasifikator biner.
- `models/gru_attention_reader.keras` — pembaca bobot attention.
- `outputs/reports/hasil_validasi_gru_attention.json` — metrik dan audit.
- `outputs/reports/gru_attention_training_history.csv` — kurva per epoch.
- `outputs/reports/classification_report_gru_validation.txt` — laporan kelas.
- `outputs/figures/kurva_training_gru_attention.png` — grafik pelatihan.
- `outputs/figures/confusion_matrix_gru_validation.png` — confusion matrix.

Jangan publikasi data SMS mentah. Berkas model dan laporan lokal tetap diabaikan
oleh `.gitignore`; gambar dapat terlihat oleh Git dan perlu diperiksa sebelum commit.

## Bukti untuk dinilai

- Training sukses dan model yang disimpan dapat dimuat lagi tanpa perubahan prediksi.
- Attention valid, bobot berjumlah satu dan mengabaikan posisi padding.
- Metrik accuracy, precision, recall, F1-score spam dan average precision.
- Data validation 516 SMS; test tetap terkunci.
- Hasil awal ini akan dipakai untuk membandingkan konfigurasi pada tuning dan ablasi.

Interpretabilitas berbasis bobot attention **bukan bukti kausal** bahwa token
tertentu menyebabkan keputusan model; visualisasi dan analisis kesalahan akan
dibahas secara hati-hati dalam tahap berikutnya.
