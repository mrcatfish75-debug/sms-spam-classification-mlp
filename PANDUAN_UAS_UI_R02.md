# UAS Tahap 6 — UI Refinement R02

**Proyek:** Klasifikasi SMS Spam — UCI SMS Spam Collection (bahasa Inggris)  
**Arah desain:** Human-Centered ML Workbench  
**Cabang kerja:** `uas-development`  
**Lokasi proyek:** `D:\1. PROJEk DL\SMS-Spam-Indonesia`

## 1. Perubahan dan batasan yang dijaga

Paket R02 hanya mengganti lapisan antarmuka. Daftar file:

| File | Tindakan |
| --- | --- |
| `app.py` | Mengganti UI Gradio versi sebelumnya |
| `ui/uas_theme.css` | Menambahkan sistem desain dan tata letak responsif |
| `tests/test_stage6_ui_r02.py` | Menambahkan delapan tes logika UI |
| `PANDUAN_UAS_UI_R02.md` | Menambahkan panduan pemasangan dan pemeriksaan |

**Tidak ada** file model, dataset, laporan eksperimen, tokenizer, atau
`src/uas_inference.py` dalam paket ini. Script Tahap 1–5 juga tidak disentuh.
Model T3, MLP, parameter terlatih, hash SHA-256, preprocessing, dan **threshold
0,5 tetap persis sama**. Hasil test final yang sudah dievaluasi satu kali tidak
boleh dihitung ulang.

### Yang diperbaiki

1. Kartu hasil tidak lagi putih dengan teks pucat; permukaan dan teks memakai
   kombinasi gelap yang konsisten.
2. Susunan dua kolom input 45% dan hasil 55% dipusatkan pada desktop; saat layar
   sempit, panel ditumpuk vertikal.
3. Hasil utama memakai `Pesan normal (Ham)` atau `Terindikasi spam` dengan
   penjelasan singkat bagi pengguna umum.
4. Probabilitas spam ditulis besar, diikuti bilah pengukuran dan **garis
   keputusan 50%**. Angka ini tidak diklaim sebagai jaminan keamanan.
5. Dua tombol `Pesan biasa` dan `Pesan promosi` menggantikan label `Examples`.
6. Pengaturan Attention dan rincian teknis berada dalam panel yang dapat dibuka.
7. Lima token Attention menjadi daftar berperingkat; angka bobot asli tetap
   terlihat. Panjang batang bersifat relatif terhadap bobot terbesar.
8. Perbandingan menampilkan dua kartu setara dan menandai jika label berbeda,
   tanpa mengklaim bahwa model dengan probabilitas tertentu selalu lebih baik.

## 2. Pasang secara aman di Windows PowerShell 5.1

**Hentikan server Gradio yang masih berjalan dengan `Ctrl+C` sebelum menyalin.**

Jalankan di PowerShell:

```powershell
Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
git branch --show-current
git status --short

$project = (Get-Location).Path
$zipPath = Join-Path $env:USERPROFILE 'Downloads\UAS_Tahap6_UI_Refinement_R02.zip'
Test-Path -LiteralPath $zipPath
Test-Path -LiteralPath (Join-Path $project 'app.py')
```

Pastikan cabang `uas-development` dan kedua `Test-Path` bernilai `True`.
**Jangan lanjut jika lokasi proyek, arsip, atau `app.py` tidak sesuai.**

Ekstrak ke folder sementara **berbeda**, bukan langsung ke proyek:

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$staging = Join-Path $env:TEMP "SMS-Spam-UI-R02-$stamp"
Expand-Archive -LiteralPath $zipPath -DestinationPath $staging

Get-ChildItem -LiteralPath $staging -Recurse -File | Select-Object FullName
```

Daftar hanya boleh berisi empat berkas pada tabel bagian 1.

Cadangkan **file UI lama**, kemudian salin hanya berkas R02 yang ditentukan:

```powershell
$backup = Join-Path $project "outputs\reports\app_before_UI_R02_$stamp.py"
Copy-Item -LiteralPath (Join-Path $project 'app.py') -Destination $backup
Test-Path -LiteralPath $backup

New-Item -ItemType Directory -Path (Join-Path $project 'ui') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $project 'tests') -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $staging 'app.py') -Destination (Join-Path $project 'app.py') -Force
Copy-Item -LiteralPath (Join-Path $staging 'ui\uas_theme.css') -Destination (Join-Path $project 'ui\uas_theme.css')
Copy-Item -LiteralPath (Join-Path $staging 'tests\test_stage6_ui_r02.py') -Destination (Join-Path $project 'tests\test_stage6_ui_r02.py')
Copy-Item -LiteralPath (Join-Path $staging 'PANDUAN_UAS_UI_R02.md') -Destination (Join-Path $project 'PANDUAN_UAS_UI_R02.md')
```

Jika PowerShell menolak penyalinan salah satu file baru karena sudah ada,
**berhenti dan periksa file lama**; jangan gunakan `-Force` tanpa memastikan
isinya. Perintah `-Force` di atas dipakai hanya untuk mengganti `app.py` yang
sudah dicadangkan.

## 3. Pemeriksaan otomatis sebelum demo

R02 tidak memerlukan instalasi ulang TensorFlow atau Gradio jika Tahap 6 sudah
berjalan. Tetap gunakan `.venv` Python 3.13 proyek.

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py

.\.venv\Scripts\python.exe scripts\13_verify_app.py --check

.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_stage6_static.py' -v

.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_stage6_ui_r02.py' -v

.\.venv\Scripts\python.exe -m pip check
```

Hasil yang dituju:

- `UAS_STAGE6_STATIC_VERIFIED` (SHA-256 model, vocab 3.443, panjang 60,
  threshold 0,5).
- **7** pengujian lama PASS.
- **8** pengujian UI baru PASS.
- `pip check`: `No broken requirements found`.

Bila salah satu gagal, **jangan** menjalankan ulang training ataupun final
test. Kirimkan output error dan pertahankan cadangan UI tadi.

## 4. Jalankan demo dan cek tampilan secara manual

```powershell
.\.venv\Scripts\python.exe app.py
```

Buka `http://127.0.0.1:7860` jika browser tidak otomatis terbuka.

### Checklist 10 menit

1. **Desktop:** judul, panel input, dan panel hasil sejajar di tengah; kartu
   hasil dan angka dapat dibaca jelas (tidak ada teks putih di atas putih).
2. **Pesan biasa:** klik contoh → klik `Analisis pesan` → baca label, persentase,
   ambang 50%, dan catatan bukan jaminan keamanan.
3. **Pesan promosi:** coba contoh kedua. Jangan menganggap contoh fiktif sebagai
   metrik eksperimen atau data berlabel.
4. **Attention:** buka `Opsi interpretasi`, centang pilihan Attention, jalankan
   ulang; buka `Cara membaca hasil & rincian teknis`, periksa urutan dan bobot
   token yang ditampilkan.
5. **MLP:** pilih `MLP baseline (UTS)`; hasil MLP harus bisa ditampilkan tanpa
   membutuhkan Attention. Jika opsi Attention aktif, aplikasi menjelaskan bahwa
   fitur tersebut hanya ada pada GRU.
6. **Perbandingan:** pilih `Bandingkan dua model`; ada **dua kartu hasil**, bukan
   satu angka yang mengklaim model paling baik.
7. **Responsif:** kecilkan lebar jendela sekitar 700–800 piksel; kedua panel
   harus bertumpuk rapi, teks tidak terpotong, tombol tetap terlihat.
8. **Keyboard:** gunakan Tab untuk berpindah kontrol, Space untuk tombol/cek,
   dan cek garis fokus terlihat jelas.
9. **Tema:** lihat dalam mode terang dan gelap OS/browser; kartu hasil tidak
   boleh berubah menjadi kombinasi teks/latar berkontras rendah.
10. **Privasi:** jangan mengetik OTP, nomor asli, atau informasi pribadi.

Catat masalah UI yang masih terlihat dan kirim screenshot desktop + layar
sempit untuk penyesuaian akhir. Tampilan di browser Windows pengguna adalah
pemeriksaan visual terakhir sebelum dinyatakan siap dipresentasikan.

## 5. Commit setelah pemeriksaan lulus

Periksa status terlebih dahulu:

```powershell
git status --short
```

Jika **file Tahap 6 sebelumnya masih untracked**, sertakan juga kode inferensi
dan dependensi Tahap 6 dalam commit; jangan mengunggah UI saja sehingga
repository tidak bisa dijalankan:

```powershell
git add app.py ui/uas_theme.css tests/test_stage6_ui_r02.py PANDUAN_UAS_UI_R02.md
git add src/uas_inference.py src/__init__.py scripts/13_verify_app.py tests/test_stage6_static.py requirements-uas-tahap6.txt PANDUAN_UAS_TAHAP6.md

git diff --cached --check
git --no-pager diff --cached --stat

git commit -m "UAS: refine Gradio inference UI and accessibility"
git push origin uas-development
```

**Jangan** `git add models/`, `data/`, `outputs/reports/`, atau
`uas_final_test_once.lock`. Bobot model dan dataset tetap ditangani menurut
protokol distribusi dan penyimpanan sebelumnya.

## 6. Pemulihan jika UI baru bermasalah

Hentikan server dengan `Ctrl+C`, kemudian salin kembali `app.py` dari nilai
variabel `$backup` yang sudah dibuat pada bagian 2:

```powershell
Copy-Item -LiteralPath $backup -Destination (Join-Path $project 'app.py') -Force
```

Rollback hanya untuk antarmuka, bukan model atau hasil eksperimen. File CSS
dan tes R02 tidak perlu dihapus untuk menjalankan app versi lama.

## 7. Jejak pemeriksaan pembuatan paket

- Gradio yang dipakai untuk uji build: **6.32.0**.
- Uji build konfigurasi antarmuka: PASS.
- Uji statis artefak beku: PASS.
- Uji regresi: **15 PASS** (7 tes pemrosesan asli + 8 tes UI R02).
- Perhitungan kontras warna: teks utama pada kartu 12,80:1; teks sekunder
  7,99:1; tombol biru dengan teks putih 4,59:1. Ini pengujian nilai warna,
  **bukan sertifikasi aksesibilitas lengkap**.
- Pemeriksaan screenshot otomatis Chromium di lingkungan pembuatan terhalang
  pembatasan sandbox browser. Karena itu hasil visual di laptop pengguna tetap
  wajib dicek lewat checklist bagian 4.
- Tidak ada train/validation/test CSV yang dibaca dan tidak ada prediksi final
  test yang diulang selama pembuatan paket.

**Kredit pendekatan:** prinsip refinement UI diadaptasi dari *Unslop*,
Damir Sokolovsky ([@heydamir](https://x.com/heydamir)), lisensi CC BY 4.0.
Dokumentasi resmi Gradio 6:
https://gradio.app/guides/gradio-6-migration-guide
