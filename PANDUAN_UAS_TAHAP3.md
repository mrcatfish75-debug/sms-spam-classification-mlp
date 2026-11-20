# UAS Tahap 3 — Hyperparameter Tuning GRU + Attention

## Tujuan dan aturan eksperimen

Membandingkan **empat konfigurasi** GRU + Attention di data **VALIDATION**.
T1 menggunakan model baseline Tahap 2 yang telah dilatih dan diverifikasi,
**tanpa mengulang training**. T2–T4 dilatih satu per satu dalam proses Python
terpisah untuk membatasi penggunaan RAM pada laptop CPU/8 GB.

| ID | Unit GRU | Dropout | Sumber |
| --- | ---: | ---: | --- |
| T1 | 32 | 0.20 | Baseline Tahap 2 (reuse) |
| T2 | 32 | 0.40 | Training baru |
| T3 | 64 | 0.20 | Training baru |
| T4 | 64 | 0.40 | Training baru |

Seluruh eksperimen mempertahankan dataset, urutan split, tokenisasi, vocabulary,
Word2Vec train-only, sequence length 60, dimensi embedding 64 (trainable),
AdditiveAttention, Dense16 ReLU, Adam(0.001), batch 32, maksimum 15 epoch,
EarlyStopping(val_loss, patience=3, restore_best_weights=True), seed 42,
dan ambang prediksi 0.5. Hanya GRU units dan dropout yang diubah.

**Kriteria pilihan ditentukan sebelum melihat hasil:** urutkan berdasarkan
F1 spam VALIDATION tertinggi, lalu recall spam, average precision (PR-AUC),
jumlah parameter paling sedikit, dan ID eksperimen. Kesetaraan dibaca dari
nilai metrik yang disimpan (6 angka desimal).

> **TEST TETAP TERKUNCI.** Script ini tidak membaca `sms_test.csv` dan tidak
> menggunakan data test untuk memutuskan konfigurasi. Hasil UAS Tahap 3
> belum berarti peningkatan performa pada data test atau data lapangan.

## File paket

- `scripts/08_tune_gru.py`: validasi input, tuning T2–T4, ranking, visualisasi,
  pemilihan bobot model, serta pemeriksaan ulang model terpilih.
- `requirements-uas-tahap3.txt`: mengacu ke dependensi Tahap 2 yang sudah ada.
- `PANDUAN_UAS_TAHAP3.md`: panduan ini.

Ekstraksi ZIP dilakukan di **akar folder proyek**, sehingga folder `scripts`
baru akan menyatu dengan `scripts` sebelumnya. Paket **tidak berisi** dan
**tidak menimpa** kode/data/model Tahap 1–2.

## Langkah menjalankan (Windows PowerShell 5.1)

1. Pindah ke proyek, periksa branch dan kondisi Git:

   ```powershell
   Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
   git branch --show-current
   git status
   ```

   Harus berada di branch `uas-development` dan bersih sebelum memasukkan ZIP.

2. Unduh ZIP dan simpan di `Downloads`, lalu:

   ```powershell
   $zipPath = Join-Path $env:USERPROFILE 'Downloads\UAS_Tahap3_Hyperparameter_Tuning.zip'
   Test-Path -LiteralPath $zipPath
   Expand-Archive -LiteralPath $zipPath -DestinationPath (Get-Location).Path
   ```

3. Periksa kelengkapan:

   ```powershell
   Test-Path 'scripts\08_tune_gru.py'
   Test-Path 'requirements-uas-tahap3.txt'
   Test-Path 'PANDUAN_UAS_TAHAP3.md'
   Test-Path 'models\gru_attention_baseline.keras'
   Test-Path 'outputs\reports\hasil_validasi_gru_attention.json'
   ```

   Kelimanya harus `True`.

4. Gunakan `.venv` Python 3.13 milik proyek. Library yang digunakan masih
   sama; instal bila perlu:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap3.txt
   ```

5. **Jalankan hyperparameter tuning**:

   ```powershell
   .\.venv\Scripts\python.exe scripts\08_tune_gru.py
   ```

   Script memvalidasi manifest dan data terhadap Tahap 1–2, menggunakan T1
   sebagai baseline, lalu melatih T2/T3/T4 **sekuensial**. Semua konfigurasi
   mengevaluasi 516 SMS validation yang sama. Proses wajar memerlukan beberapa
   menit pada CPU. Jangan tutup PowerShell saat training sedang berlangsung.

6. Periksa laporan seleksi:

   ```powershell
   $r = Get-Content 'outputs\reports\uas_tuning_selection.json' -Raw | ConvertFrom-Json
   $r.status
   $r.winner_trial
   $r.winner_config | Format-List
   $r.winner_validation_metrics | Format-List
   $r.test_status
   Import-Csv 'outputs\reports\uas_tuning_grid_results.csv' | Format-Table -AutoSize
   Test-Path 'models\gru_attention_tuned.keras'
   Test-Path 'models\gru_attention_tuned_reader.keras'
   Test-Path 'outputs\figures\tuning_gru_validation.png'
   git status --short
   ```

   `status` harus `UAS_TUNING_VALIDATION_ONLY_PASS`, `test_status` harus
   `LOCKED_NOT_EVALUATED`, dan ketiga `Test-Path` harus `True`.

## Output

| Path | Fungsi |
| --- | --- |
| `outputs/reports/uas_tuning_selection.json` | Status, seluruh konfigurasi, rule seleksi, pemenang, metrik, checksum |
| `outputs/reports/uas_tuning_grid_results.csv` | Tabel empat model untuk laporan UAS |
| `outputs/reports/uas_tuning/T2_validation.json` dan T3/T4 | Audit tiap eksperimen |
| `outputs/reports/uas_tuning/T2_training_history.csv` dan T3/T4 | Log epoch tiap model |
| `outputs/figures/tuning_gru_validation.png` | Grafik perbandingan metrik validation |
| `models/uas_tuning/T2_classifier.keras` dan T3/T4 | Bobot kandidat tiap konfigurasi |
| `models/uas_tuning/T2_attention_reader.keras` dan T3/T4 | Pembaca attention per konfigurasi |
| `models/gru_attention_tuned.keras` | Salinan bobot model terpilih, tanpa menghapus baseline Tahap 2 |
| `models/gru_attention_tuned_reader.keras` | Pembaca attention untuk model terpilih |

> Nama file `tuned` **tidak menjamin** T2/T3/T4 lebih bagus. Bila T1 memiliki
> metrik terbaik, script memilih T1 dan menyalin kembali bobot baseline
> Tahap 2; ini hasil ilmiah yang sah.

## Jika terhenti atau ingin mengulang

- Jalankan perintah utama kembali: trial yang memiliki laporan dan checksum
  lengkap akan **dilewati** sehingga tidak perlu mengulang semuanya.
- Bila file trial tidak cocok/terpotong, jalankan kembali dengan `--force`
  untuk melatih ulang T2–T4:

  ```powershell
  .\.venv\Scripts\python.exe scripts\08_tune_gru.py --force
  ```

- Jangan menjalankan ulang Tahap 1 atau Tahap 2 ketika tuning masih berjalan.
  Bila file input diubah, checksum akan berubah dan eksperimen lama ditolak.

## Etika dan Git

- Output model, dataset mentah, `.npz` dan laporan JSON memiliki aturan
  `.gitignore`. Jangan menghapus aturan privasi tersebut.
- Yang akan di-*commit* **setelah hasil diperiksa bersama** adalah file
  `scripts/08_tune_gru.py`, `requirements-uas-tahap3.txt`,
  `PANDUAN_UAS_TAHAP3.md`, dan grafik `tuning_gru_validation.png`.
- **Belum** perlu push atau commit sebelum laporan `PASS` diverifikasi.
- Tahap berikutnya adalah ablation serta interpretabilitas attention dan
  analisis lima kasus salah prediksi. Model final diuji pada test terpisah
  hanya setelah metode seleksi dan evaluasi selesai ditetapkan.
