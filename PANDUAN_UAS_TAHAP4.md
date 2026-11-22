# UAS Tahap 4 — Ablation Study, Analisis Kesalahan, dan Attention

## 1. Prasyarat dan batas eksperimen

Paket Tahap 4 **menggunakan hasil Tahap 1–3 yang sudah ada**. Jangan ulangi
tokenisasi, Word2Vec, split, training GRU, ataupun tuning. Model referensi adalah
**T3** terpilih pada Tahap 3: Word2Vec (train-only) → Embedding 64 trainable →
GRU 64 → AdditiveAttention → Dense 16 ReLU → Dropout 0,20 → Sigmoid.

Prasyarat wajib:

- Branch Git: `uas-development`.
- `scripts/06_prepare_sequences.py`, `07_train_gru.py`, `08_tune_gru.py`.
- `data/processed/sequences_train.npz` dan `sequences_validation.npz`.
- `data/splits/sms_train.csv` dan `sms_validation.csv`.
- `models/word2vec_embedding_matrix.npy` dan vocabulary/config yang sama.
- `models/gru_attention_tuned.keras`, `gru_attention_tuned_reader.keras`.
- `outputs/reports/uas_tuning_selection.json` berstatus PASS, pemenang T3.

Seluruh script memverifikasi kembali **checksum input dan model**, seed, split,
ukuran vocabulary, konfigurasi T3, serta status TEST yang masih terkunci.

> Kedua script **tidak membaca** `data/splits/sms_test.csv` dan **tidak**
> memilih hyperparameter baru. Seluruh hasil di Tahap 4 adalah VALIDATION.

## 2. Ablation study yang adil

| Varian | Inisialisasi embedding | GRU | Attention | Fungsi |
| --- | --- | --- | --- | --- |
| T3 Full | Word2Vec train-only | 64 | Ya | Model kontrol, reuse Tahap 3 |
| A1 — No Attention | Word2Vec train-only | 64 | Tidak | Mengukur efek melepas attention |
| A2 — No Word2Vec Init | Random uniform seed 42 | 64 | Ya | Mengukur efek inisialisasi Word2Vec |

**Catatan metodologis:** A1 memakai hidden state terakhir GRU sebagai pengganti
context attention (dimensi tetap 64). A2 tetap **mempelajari** embedding selama
training, tetapi bobot awalnya bukan Word2Vec. Jadi A2 adalah ablasi pada
**inisialisasi Word2Vec**, bukan penghapusan seluruh embedding.

Semua varian memakai train/validation yang sama, panjang sekuensial 60,
vocabulary yang sama, embedding trainable, Dense16, Dropout0.20, Adam(0.001),
batch32, seed42, max15 epoch, dan EarlyStopping(val_loss,patience3,
restore_best_weights=True). Setiap varian baru dilatih dalam proses Python
terpisah agar RAM laptop 8 GB tidak terus menumpuk. Model T3 **tidak dilatih ulang**.

Hasil selisih metrik menggambarkan sensitivitas terhadap penggantian satu
komponen pada split dan seed tersebut, bukan bukti efek kausal yang berlaku
secara universal. Bila ablasi melampaui model lengkap pada sebagian metrik,
catat apa adanya; tidak boleh memilih/mengganti T3 menggunakan TEST.

## 3. Isi paket

- `scripts/09_ablation_gru.py`: train dua ablasi (sekuensial, checkpoint dan resume),
  validasi model, bandingkan F1 spam serta PR-AUC dengan model T3.
- `scripts/10_interpret_validation.py`: analisis keseluruhan validation,
  lima contoh kasus, pemeriksaan token dan bobot attention, serta form review.
- `requirements-uas-tahap4.txt`: menggunakan dependensi Tahap 3 tanpa upgrade wajib.
- `PANDUAN_UAS_TAHAP4.md`: panduan dan protokol laporan.

ZIP hanya berisi empat berkas tersebut. Tidak berisi dataset, bobot lama,
atau skrip lama yang bisa menimpa hasil UTS/Tahap 1–3.

## 4. Instalasi Windows PowerShell 5.1

1. Buka folder proyek dan pastikan branch serta Git bersih:

   ```powershell
   Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
   git branch --show-current
   git status
   ```

2. Unduh `UAS_Tahap4_Ablation_Interpretability.zip` ke folder Downloads:

   ```powershell
   $zipPath = Join-Path $env:USERPROFILE 'Downloads\UAS_Tahap4_Ablation_Interpretability.zip'
   Test-Path -LiteralPath $zipPath
   Expand-Archive -LiteralPath $zipPath -DestinationPath (Get-Location).Path
   ```

3. Verifikasi:

   ```powershell
   Test-Path 'scripts\09_ablation_gru.py'
   Test-Path 'scripts\10_interpret_validation.py'
   Test-Path 'requirements-uas-tahap4.txt'
   Test-Path 'PANDUAN_UAS_TAHAP4.md'
   Test-Path 'models\gru_attention_tuned.keras'
   Test-Path 'outputs\reports\uas_tuning_selection.json'
   ```

   Keenam baris harus `True`.

4. Dependensi (biasanya sudah tersedia):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap4.txt
   ```

## 5. Menjalankan Tahap 4A — Ablation Study

```powershell
.\.venv\Scripts\python.exe scripts\09_ablation_gru.py
```

Tunggu dua varian selesai. Jangan jalankan training lain secara bersamaan.
Bila proses terhenti, perintah yang sama akan melewati varian yang telah PASS
dan checksum-nya masih valid. Jika sengaja ingin melatih ulang, gunakan
`scripts\09_ablation_gru.py --force` (mengulang **dua** varian).

Periksa:

```powershell
$a = Get-Content 'outputs\reports\uas_ablation_summary.json' -Raw | ConvertFrom-Json
$a.status
$a.control_trial
$a.test_status
Import-Csv 'outputs\reports\uas_ablation_comparison.csv' | Format-Table -AutoSize
Test-Path 'outputs\figures\ablation_gru_validation.png'
```

Status yang diharapkan:

- `UAS_ABLATION_VALIDATION_ONLY_PASS`
- `T3`
- `LOCKED_NOT_EVALUATED`
- `True` untuk grafik.

## 6. Menjalankan Tahap 4B — Analisis Validasi dan Attention

**Setelah Tahap 4A PASS**, jalankan:

```powershell
.\.venv\Scripts\python.exe scripts\10_interpret_validation.py
```

Script memilih **semua kasus salah** (pada laporan Tahap 3 terdapat 2), lalu
menambahkan pesan yang sebenarnya **diklasifikasi benar** tetapi memiliki
margin probabilitas terkecil terhadap threshold 0,5 sampai berjumlah **5 kasus**.
Kedua jenis kasus **tidak boleh digabung menjadi lima kesalahan**.

Periksa:

```powershell
$b = Get-Content 'outputs\reports\uas_validation_attention_analysis.json' -Raw | ConvertFrom-Json
$b.status
$b.errors
$b.false_positives
$b.false_negatives
$b.genuine_misclassifications_in_report
$b.correct_lowest_margin_in_report
$b.test_status
Test-Path 'outputs\reports\uas_validation_manual_review.csv'
Test-Path 'outputs\figures\attention_validation_cases.png'
```

Status yang diharapkan: `UAS_INTERPRET_VALIDATION_ONLY_PASS`, `test_status`
`LOCKED_NOT_EVALUATED`, dan `True` untuk dua file output.

## 7. Review manual yang wajib dilakukan anggota

Buka `outputs/reports/uas_validation_manual_review.csv` di VS Code atau
Excel; isi tiga kolom **untuk masing-masing lima contoh**:

- `review_label_valid_ya_tidak`: label pada dataset masih sesuai konteks?
- `dugaan_penyebab_manual`: alasan yang *masuk akal* jika salah, atau alasan
  kasus benar tetapi tidak terlalu yakin; hindari menyatakan sebab pasti.
- `catatan_reviewer`: batas konteks, singkatan, pola SMS, teks bermasalah, dll.

Sebagian identitas seperti nomor telepon, angka, URL, dan email disamarkan,
tetapi anonimisasi ini **tidak sempurna**: **CSV dan JSON laporan tetap
bersifat internal dan tidak boleh dipublikasikan**. Jangan salin SMS asli ke
gambar atau repositori publik tanpa audit privasi.

Bobot attention memperlihatkan distribusi fokus model terhadap **token yang
masuk**; bobot tertinggi bukan bukti bahwa kata tersebut menyebabkan hasil
prediksi. Grafik publik hanya memperlihatkan posisi token, **bukan isi SMS**.

## 8. File keluaran

| File | Isi |
| --- | --- |
| `outputs/reports/uas_ablation_summary.json` | Perbandingan T3 lengkap vs A1/A2 dan metadata validasi |
| `outputs/reports/uas_ablation_comparison.csv` | Tabel metrik, selisih F1 & PR-AUC |
| `outputs/reports/uas_ablation/*_validation.json` | Bukti dua training ablation, checksum model |
| `models/uas_ablation/*.keras` | Model A1 dan A2 (tidak menimpa model T3) |
| `outputs/figures/ablation_gru_validation.png` | Grafik F1 spam dan PR-AUC per varian |
| `outputs/reports/uas_validation_attention_analysis.json` | Rekap validation, dua jenis kasus, bobot attention teratas |
| `outputs/reports/uas_validation_manual_review.csv` | Lima kasus dengan kolom review manual |
| `outputs/figures/attention_validation_cases.png` | Heatmap berbasis posisi token tanpa teks mentah |

## 9. Git setelah diperiksa bersama

Jangan commit/push dulu sebelum semua hasil dibaca bersama. Kelak yang
layak masuk Git adalah dua skrip, file requirements, panduan, dan dua PNG
tanpa teks SMS. Data, `.npz`, file `.keras`, dan isi `outputs/reports/` tetap
diabaikan `.gitignore`.

Langkah selanjutnya setelah Tahap 4 + review manual: tetapkan protokol final
evaluasi; jalankan TEST **hanya sekali** untuk model yang sudah dipilih,
baru siapkan aplikasi web sederhana Gradio, laporan UAS, poster, dan demo.
