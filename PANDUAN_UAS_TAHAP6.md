# UAS Tahap 6 — Gradio Web Application (Inference Only)

Proyek: **Klasifikasi SMS Spam (UCI SMS Spam Collection, bahasa Inggris)**  
Cabang Git: `uas-development`  
Folder Windows: `D:\1. PROJEk DL\SMS-Spam-Indonesia`

## A. Cakupan dan perlindungan hasil penelitian

Paket ini **menambahkan** file `app.py`, `src/uas_inference.py`, `src/__init__.py`,
`scripts/13_verify_app.py`, `tests/test_stage6_static.py`,
`requirements-uas-tahap6.txt`, dan panduan ini.

- Tidak menimpa script Tahap 1–5, model, tokenisasi, dataset, atau hasil penelitian.
- Tidak pernah memanggil `fit()`, melakukan tuning, membuka `sms_test.csv`, atau
  menghitung ulang hasil final test.
- Threshold **0,5**, model UTS **MLP**, dan model UAS **T3** ditetapkan dari
  `outputs/reports/uas_final_protocol_lock.json`.
- Setiap artefak model/TF-IDF/vocabulary diperiksa SHA-256 terhadap freeze.
- Hasil probabilitas untuk satu SMS diberikan sebagai *inferensi demo*, bukan
  metrik baru atau evaluasi ilmiah.
- Input SMS tidak ditulis oleh kode aplikasi; jangan mengetik OTP atau data
  pribadi, khususnya apabila aplikasi kelak dipublikasikan.

## B. Pasang file (Windows PowerShell 5.1)

**Lakukan di proyek asli, bukan folder backup.** Pastikan sebelumnya sudah ada
snapshot backup lengkap Tahap 1–5.

```powershell
Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'

git branch --show-current
git status --short

$zipPath = Join-Path $env:USERPROFILE 'Downloads\UAS_Tahap6_Gradio_App.zip'
Test-Path -LiteralPath $zipPath

Expand-Archive -LiteralPath $zipPath -DestinationPath (Get-Location).Path

Test-Path 'app.py'
Test-Path 'src\uas_inference.py'
Test-Path 'scripts\13_verify_app.py'
Test-Path 'requirements-uas-tahap6.txt'
```

Semua `Test-Path` terakhir harus `True`. `Expand-Archive` sengaja dijalankan
tanpa `-Force`: jika ada file dengan nama sama, **STOP** dan periksa isinya.

## C. Instal paket Gradio

Gunakan `.venv` Python 3.13 yang selama ini dipakai (tidak perlu membuat ulang):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap6.txt

.\.venv\Scripts\python.exe -m pip check

.\.venv\Scripts\python.exe -c "import gradio, tensorflow; print('Gradio:', gradio.__version__, '| TensorFlow:', tensorflow.__version__)"
```

Gradio 6 digunakan. `tensorflow==2.21.0` dipertahankan melalui
`requirements-tahap2.txt` yang telah dipakai pada UTS/UAS.

## D. Verifikasi read-only, TANPA TensorFlow dan TANPA dataset

```powershell
.\.venv\Scripts\python.exe scripts\13_verify_app.py --check

.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_stage6_static.py' -v
```

Target:

```
Freeze T3/MLP SHA256: PASS
Tokenizer 3443 vocabulary / 60 tokens: PASS
Threshold 0.5: PASS
STATUS: UAS_STAGE6_STATIC_VERIFIED
NO TRAIN/VALIDATION/TEST DATASET WAS READ
```

Jika gagal, **jangan** membuat ulang model, mengubah tokenizer, atau menghapus
`uas_final_protocol_lock.json`. Kirim log kesalahan untuk diperiksa.

## E. Smoke test model asli menggunakan contoh buatan

```powershell
.\.venv\Scripts\python.exe scripts\13_verify_app.py --smoke
```

Script hanya mencoba dua pesan fiktif yang tertulis di kode, bukan mengambil
pesan dari UCI ataupun test split. Yang diuji:

- MLP dapat memuat `.keras` + TF-IDF yang dibekukan.
- GRU T3 dapat memuat `.keras` + vocabulary asli.
- Output probabilitas berada pada rentang 0–1.
- Attention reader mengembalikan bobot valid dan padding bernilai nol.

Target status: `UAS_STAGE6_SMOKE_PASS`.

Laporan baru (khusus demo, **bukan** evaluasi test):
`outputs/reports/uas_stage6_app_smoke.json`.

Angka probabilitas untuk contoh fiktif **tidak memiliki label ground truth**.
Jangan menghitung accuracy atau menyatakan eksperimen baru dari smoke test ini.

## F. Jalankan aplikasi lokal

```powershell
.\.venv\Scripts\python.exe app.py
```

Browser biasanya membuka `http://127.0.0.1:7860` secara otomatis. Jika tidak,
buka alamat tersebut secara manual.

Urutan demo:

1. Ketik satu SMS berbahasa Inggris.
2. Pilih `GRU T3 (UAS)` dan klik **Analisis SMS**.
3. Perhatikan Spam/Ham, probabilitas, dan jumlah token tak dikenal.
4. Aktifkan bobot Attention untuk menjelaskan interpretabilitas secara hati-hati.
5. Pilih `Bandingkan keduanya` untuk menampilkan keluaran GRU T3 dan MLP UTS.

Tekan `Ctrl+C` di terminal untuk menghentikan server.

### Apa yang TIDAK boleh dilakukan

- Jangan menjalankan ulang `scripts/12_evaluate_final_test.py --evaluate-test`.
- Jangan menghapus `outputs/reports/uas_final_test_once.lock`.
- Jangan menjalankan ulang training atau tuning untuk "memperbaiki" angka test.
- Jangan mengganti model T3 dengan model lain berdasarkan hasil test.
- Jangan menambahkan folder `data/` atau CSV review yang memuat SMS pribadi ke Space publik.

## G. Komit setelah uji lokal benar-benar PASS

```powershell
git status --short

git add app.py src/__init__.py src/uas_inference.py scripts/13_verify_app.py tests/test_stage6_static.py requirements-uas-tahap6.txt PANDUAN_UAS_TAHAP6.md

git diff --cached --check
git --no-pager diff --cached --stat

git commit -m "UAS: add frozen-model Gradio inference demo"
git push origin uas-development
```

`models/`, dataset, dan `outputs/reports/` **tidak** perlu dipaksa masuk Git.
Tahap berikutnya adalah menyiapkan distribusi bobot model melalui unduhan
terpisah/Hugging Face Hub, lalu paket Spaces yang tidak menyertakan data SMS.

## H. Kriteria keberhasilan Tahap 6

| Pemeriksaan | Target |
| --- | --- |
| Freeze + semua SHA-256 | PASS |
| Vocab/sequence/threshold | 3443 / 60 / 0,5 |
| Model T3 | `gru_attention_tuned.keras`, 246.433 parameter |
| Model MLP | `mlp_baseline.keras`, 130.177 parameter |
| Smoke test fiktif | PASS, tanpa dataset |
| Aplikasi lokal | Browser dapat menampilkan probabilitas |
| Attention | Hanya GRU T3, bobot ter-mask normal |
| Komit GitHub | Dilakukan setelah uji lokal |

## I. Catatan interpretasi untuk presentasi UAS

Validasi T3 memang lebih tinggi daripada MLP pada F1-score spam, tetapi test
final menunjukkan MLP sedikit lebih tinggi pada F1-score spam. Kedua model
harus dipresentasikan secara jujur. Demo ini memperlihatkan kemampuan
*inference* dan interpretabilitas, **bukan bukti** bahwa GRU selalu lebih baik.

Referensi teknis API:

- Gradio Blocks: https://gradio.app/docs/gradio/blocks
- Gradio 6 migration: https://gradio.app/guides/gradio-6-migration-guide
- Hugging Face Spaces: https://huggingface.co/docs/hub/spaces-sdks-gradio
