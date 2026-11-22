# UAS Tahap 5 — Audit Lima Kesalahan + Evaluasi Final

Proyek: **SMS Spam Classification — UCI SMS Spam Collection**  
Cabang Git: `uas-development`  
Folder Windows: `D:\1. PROJEk DL\SMS-Spam-Indonesia`  
Sasaran perangkat: Windows PowerShell 5.1, Python 3.13, CPU, RAM 8 GB.

## 1. Kebijakan keselamatan dan status

Paket ini **menambah dua skrip** dan tidak mengubah skrip, data, atau model Tahap 1–4. Semua output baru masuk `outputs/reports/` atau `outputs/figures/`. File sensitif di `outputs/reports/` tetap diabaikan Git melalui `.gitignore` lama.

- **Tahap 5A:** 100% TRAIN/VALIDATION saja. Tidak membaca isi `sms_test.csv`, tidak melakukan training. Membuat lima kasus **yang benar-benar salah diprediksi oleh MLP UTS**, kemudian mengunci protokol setelah dua anggota mengisinya.
- **Tahap 5B:** Evaluasi TEST satu kali, hanya setelah Tahap 5A `FROZEN` dan persetujuan eksplisit. Dua model yang sudah ada (MLP UTS dan T3 GRU UAS) menggunakan **test yang sama**. Tanpa fit, tuning, atau resampling.
- **Jangan menjalankan** ulang `03_split_tfidf.py`, `04_train_mlp.py`, `06_prepare_sequences.py`, `07_train_gru.py`, `08_tune_gru.py`, `09_ablation_gru.py`, atau `10_interpret_validation.py` saat menyiapkan hasil final. Beberapa skrip tersebut dapat menimpa model/laporan/review sebelumnya.

### Ketentuan lima kesalahan dari dosen

Tahap 4B hanya memiliki **2 kesalahan T3** (`V1`, `V2`) dan **3 prediksi benar margin rendah** (`V3`–`V5`). Jangan menyebut V3–V5 sebagai kesalahan. Tahap 5A merekonstruksi **5 kesalahan MLP pada validation (3 FP + 2 FN)**, kemudian mencatat dua kesalahan T3 terpisah. Dalam laporan, selalu sebut **model sumber kasusnya**. Jika dosen secara khusus mewajibkan lima kesalahan *model T3*, mintakan klarifikasi; jangan memanipulasi hasil.

## 2. Pemasangan paket

Pindahkan ZIP ke `Downloads`, kemudian buka PowerShell:

```powershell
Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
git branch --show-current
git status --short
```

Pastikan branch `uas-development` dan worktree bersih. **Jangan memasang pada cabang `main`**.

```powershell
$zipPath = Join-Path $env:USERPROFILE 'Downloads\UAS_Tahap5_Final_Evaluation.zip'
Test-Path -LiteralPath $zipPath
Expand-Archive -LiteralPath $zipPath -DestinationPath (Get-Location).Path
```

Hasil `Test-Path` harus `True`. Tidak memakai `-Force` karena ZIP ini hanya berisi nama file baru.

```powershell
Test-Path 'scripts\11_prepare_final_evaluation.py'
Test-Path 'scripts\12_evaluate_final_test.py'
Test-Path 'requirements-uas-tahap5.txt'
Test-Path 'PANDUAN_UAS_TAHAP5.md'
```

Semuanya harus `True`.

Opsional untuk instalasi jika dependensi lama belum ada:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap5.txt
```

Tidak perlu upgrade Python dan tidak perlu memasang GPU/CUDA.

## 3. Tahap 5A-1 — Temukan tepat lima kesalahan MLP

Sebelum menjalankan, **pastikan** hasil manual review T3 yang sudah disetujui tetap ada:

```powershell
.\.venv\Scripts\python.exe -c "import pandas as pd; d=pd.read_csv('outputs/reports/uas_validation_manual_review.csv'); print('T3 review:',len(d),'kasus; catatan kosong:',d['catatan_reviewer'].isna().sum())"
```

Hasil diharapkan `5 kasus; catatan kosong: 0`.

Lalu jalankan **hanya** perintah ini:

```powershell
.\.venv\Scripts\python.exe scripts\11_prepare_final_evaluation.py --prepare
```

Hasil yang benar:

- `MLP UTS: 5 kesalahan (3 FP, 2 FN)`.
- `GRU T3: 2 kesalahan (0 FP, 2 FN)`.
- Metrik validation sama dengan laporan model yang sudah tersimpan.
- `Review: outputs/reports/uas_mlp_five_errors_manual_review.csv`.
- `TEST TIDAK DIBACA`.

Skrip **tidak menghapus/menimpa CSV manual lima kasus jika sudah ada**. Jika file sudah ada tetapi kolom otomatis tidak cocok, skrip berhenti.

### Lihat lima kasus yang memang salah

```powershell
Import-Csv 'outputs\reports\uas_mlp_five_errors_manual_review.csv' |
Select-Object case_id, source_row, jenis_kasus, label_aktual, prediksi, probabilitas_spam |
Format-Table -AutoSize
```

Untuk memeriksa teks yang disamarkan dan kolom review:

```powershell
code 'outputs\reports\uas_mlp_five_errors_manual_review.csv'
```

**STOP DI SINI dan kirimkan hasil `--prepare` serta lima kasus tersebut ke saya untuk review manual.** Jangan membuka data test atau melakukan freeze dahulu.

## 4. Tahap 5A-2 — Isi catatan manusia dan kunci protokol

Ini langkah **setelah** kedua anggota menyetujui analisis lima kesalahan.

Dalam `outputs/reports/uas_mlp_five_errors_manual_review.csv`, hanya isi tiga kolom berikut:

1. `review_label_valid_ya_tidak`: `YA`, `TIDAK`, atau kosong jika tidak dapat diverifikasi.
2. `dugaan_penyebab_manual`: dugaan yang jelas ditandai sebagai hipotesis.
3. `catatan_reviewer`: alasan validitas label, batasan bukti, serta jenis kesalahan.

Jangan mengganti `source_row`, isi SMS, probabilitas, label, atau prediksi. Jangan klaim token `[NUM]` dipakai saat training; itu hanya penyamaran **laporan**. Jangan menjelaskan bobot attention sebagai bukti sebab-akibat.

Sebelum freeze, cadangkan berkas review yang sudah disepakati:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Copy-Item -LiteralPath 'outputs\reports\uas_mlp_five_errors_manual_review.csv' -Destination "outputs\reports\uas_mlp_five_errors_review_backup_$stamp.csv"
```

Kemudian, bila tidak ada kolom dugaan/catatan kosong:

```powershell
.\.venv\Scripts\python.exe scripts\11_prepare_final_evaluation.py --freeze --confirm FREEZE_UTS_UAS
```

Hasil yang diharapkan:

```
UAS TAHAP 5A - FINAL EVALUATION PROTOCOL FROZEN
Status: UAS_FINAL_EVALUATION_PROTOCOL_FROZEN
Model dan tokenizer dicatat melalui SHA256; data TEST BELUM DIBACA.
```

Freeze hanya bisa dibuat sekali tanpa menimpa. SHA256 disimpan untuk mengunci artefak MLP, T3, TF-IDF, vocabulary, tokenizer, data train/validation, skrip dan CSV review. Test **tidak di-hash atau dibaca** pada 5A.

## 5. Tahap 5B — PENGUJIAN TEST SATU KALI (JANGAN JALANKAN DULU)

Setelah freeze, gunakan perintah status aman:

```powershell
.\.venv\Scripts\python.exe scripts\12_evaluate_final_test.py --status
```

Perintah ini hanya mengecek kelengkapan protokol dan validasi. **Tidak membuka test**.

**Tunggu persetujuan setelah saya memeriksa output 5A.** Perintah evaluasi berikut hanya dicantumkan sebagai dokumentasi dan tidak boleh dijalankan sebelum validasi akhir:

```powershell
.\.venv\Scripts\python.exe scripts\12_evaluate_final_test.py --evaluate-test --confirm EVALUATE_LOCKED_TEST_ONCE
```

Ketika benar-benar diizinkan, skrip melakukan:

1. Memeriksa ulang semua SHA256 model, preprocessing, laporan dan kode yang dibekukan.
2. Menulis `outputs/reports/uas_final_test_once.lock` **sebelum** membaca TEST.
3. Membaca 516 SMS TEST satu kali dan mengecek jumlah, distribusi, integritas sumber serta anti-leakage.
4. Melakukan `transform` TF-IDF menggunakan vectorizer tersimpan (tidak `fit`).
5. Tokenisasi 60 langkah menggunakan vocabulary Word2Vec yang sama (tidak melatih ulang).
6. Menggunakan MLP dan T3 terlatih untuk inferensi.
7. Menyimpan accuracy, precision, recall, F1 Spam, average precision (PR-AUC), dan confusion matrix untuk masing-masing model.

Output final yang nantinya dihasilkan:

```text
outputs/reports/uas_final_test_once.lock
outputs/reports/uas_final_test_results.json
outputs/reports/uas_final_test_comparison.csv
outputs/figures/uas_final_test_model_comparison.png
outputs/figures/uas_final_test_confusion_matrices.png
```

Tidak ada SMS mentah dalam JSON/CSV final. `outputs/reports/` tetap bersifat privat; dua PNG hanya berisi metrik dan angka matriks sehingga dapat ditambahkan ke GitHub setelah diperiksa.

**Jika evaluasi gagal setelah marker dibuat, JANGAN hapus marker dan JANGAN rerun otomatis.** Kirim pesan kesalahannya untuk audit; data TEST mungkin telah dibaca. Mekanisme ini sengaja melindungi aturan satu kali evaluasi.

## 6. Git setelah audit berhasil

Tahap 5 boleh di-commit setelah kedua skrip terpasang dan tahap validasi berhasil. **Jangan menambah data privat, model hasil training, review CSV, lock, atau hasil JSON ke GitHub publik.**

```powershell
git branch --show-current
git status --short
git add PANDUAN_UAS_TAHAP5.md requirements-uas-tahap5.txt scripts/11_prepare_final_evaluation.py scripts/12_evaluate_final_test.py
git --no-pager diff --cached --stat
git diff --cached --check
git commit -m "UAS: freeze evaluation protocol and prepare final test"
git push origin uas-development
git status
```

Grafik test final, setelah diperiksa dan hasilnya diterima, dapat di-commit terpisah sebagai bukti eksperimen.

## 7. Catatan laporan UAS dan pengumpulan

- Tampilkan lima kesalahan MLP dan dua kesalahan T3 **dengan asal model yang eksplisit**.
- Jangan memperlakukan tiga prediksi benar margin rendah dari Tahap 4 sebagai kesalahan.
- Kinerja akhir TEST harus ditulis apa adanya, meskipun tidak meningkat dibanding baseline.
- **UAS juga membutuhkan** refleksi etika ½–1 halaman, laporan ilmiah 12–15 halaman, minimal delapan referensi, Gradio/Streamlit, demo langsung, video 3–5 menit atau demo yang dapat diakses, poster A2, dan formulir kontribusi/peer review.
- Menurut instruksi pengumpulan terbaru, folder Google Drive mengikuti struktur repository UTS. Tautan Drive harus tercantum di hardcopy. Tenggat hardcopy **11 Desember 2026**, diletakkan di meja Bu Novera di jurusan.

Paket diuji pada salinan terpisah dengan data **sintetis** untuk jalur evaluasi satu kali; tidak ada hasil test UCI asli yang dilaporkan oleh paket ini sebelum Anda mengizinkan evaluasi di laptop sendiri.
