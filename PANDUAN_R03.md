# UAS Tahap 6B — R03 Final Polish (source-aware, fail-safe)

**Konteks:** Klasifikasi SMS Spam menggunakan MLP baseline UTS dan GRU + Attention T3 UAS. Perbaikan UI berdasarkan temuan 6A: D01–D03 stale result, C05 token sepanjang 2.001 karakter merusak daftar Attention, dan audit keamanan HTML.

## Batas kepastian

Paket ini dibuat dari bukti pengujian dan arsip model yang tersedia. **`app.py` versi R02 yang berjalan di Windows tidak terdapat dalam arsip yang tersedia di sesi pembuatan ini**, dan file tersebut juga belum ada pada branch `uas-development`. Karena itu paket menggunakan *installer source-aware* yang membaca `app.py` yang sesungguhnya di komputer. Pemasang berhenti tanpa memodifikasi aplikasi bila tidak dapat mengidentifikasi event Gradio dan komponen output secara aman. Jangan abaikan pesan `R03_PREFLIGHT_BLOCKED`.

## Yang diperbaiki ketika preflight PASS

1. Pada perubahan teks SMS, pilihan model, atau checkbox Attention, keluaran hasil lama diinvalida­si melalui event Gradio `.change` yang tidak antre (`queue=False`). Tombol contoh SMS juga memicu `.change` saat textbox berubah.
2. Output HTML dari callback Analisis dibungkus sanitizer defensif. HTML/script berbahaya dan event handler dihapus; teks berurutan tanpa spasi yang ekstrem dipendekkan **hanya pada presentasi HTML**, tidak pada token sebelum inferensi.
3. `ui/uas_theme.css` diberi tambahan gaya status *hasil perlu diperbarui* dan pembatas lebar token. CSS R02 lainnya dipertahankan.
4. File sumber inferensi, tokenizer, model `.keras`, data dan evaluator tidak diubah.

**Batas:** Invalidasi ini mengatasi keadaan normal saat pengguna mengubah teks/model. Analisis yang sedang berlangsung tidak bisa dijamin selesai berurutan hanya oleh event `.change` Gradio; karena itu wajib lakukan uji regresi R03-RACE dengan mengubah input saat analisis sedang berjalan. Jika hasil lama kembali muncul, jangan Freeze UI dan lakukan patch khusus berbasis kode R02.

**Sanitizer:** merupakan perlindungan pada output HTML, bukan audit keamanan yang komprehensif terhadap seluruh sistem. Aplikasi tetap digunakan secara lokal dan tidak menyimpan SMS sensitif.

## Cara instalasi aman — Windows PowerShell 5.1

1. Di terminal yang menjalankan Gradio, tekan **Ctrl+C** untuk menghentikan server.
2. Unduh ZIP dan ekstrak seluruh isinya langsung ke root:
   `D:\1. PROJEk DL\SMS-Spam-Indonesia`.
   Struktur paket harus menjadi `tools/apply_r03.py`, `payload/r03_helpers.py`, `tests/test_r03_helpers.py` dan sebagainya.
3. Di PowerShell:

```powershell
Set-Location -LiteralPath 'D:\1. PROJEk DL\SMS-Spam-Indonesia'
git branch --show-current
.\.venv\Scripts\python.exe tools\apply_r03.py --project . --inspect
.\.venv\Scripts\python.exe tools\apply_r03.py --project . --dry-run
```

**HANYA jika status `R03_PREFLIGHT_PASS`,** lanjutkan:

```powershell
.\.venv\Scripts\python.exe tools\apply_r03.py --project . --apply
.\.venv\Scripts\python.exe -m py_compile app.py
.\.venv\Scripts\python.exe tools\r03_verify_local.py --project .
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_r03_*.py' -v
.\.venv\Scripts\python.exe scripts\13_verify_app.py --check
.\.venv\Scripts\python.exe scripts\13_verify_app.py --smoke
.\.venv\Scripts\python.exe app.py
```

Untuk melihat tampilan R03, buka `http://127.0.0.1:7860`, lalu tekan Ctrl+Shift+R pada browser.

Jika `R03_PREFLIGHT_BLOCKED`: **tidak ada perubahan pada `app.py`**. Kirim `app.py` R02, `ui/uas_theme.css`, dan bila memungkinkan `tests/test_stage6_ui_r02.py` supaya versi patch yang cocok bisa ditulis. Jangan mengedit file model.

## Uji regresi wajib setelah R03

| ID | Langkah | PASS |
|---|---|---|
| R03-D01 | Analisis pesan promosi → ubah SMS menjadi percakapan tanpa klik Analisis | Hasil lama menjadi *perlu diperbarui* |
| R03-D02 | Analisis dengan GRU → pilih MLP tanpa analisis | Hasil lama menjadi *perlu diperbarui* |
| R03-D03 | Analisis perbandingan → pilih GRU tanpa analisis | Dua kartu lama tidak tetap menjadi hasil aktif |
| R03-INT | Analisis → aktif/nonaktifkan checkbox Attention | Hasil/penjelasan lama ditandai belum diperbarui |
| R03-EXAMPLE | Analisis → klik Pesan biasa/Promosi | Hasil lama diinvalida­si |
| R03-C05 | Input 2.001 karakter `A` → Attention aktif → Analisis | Tidak menyebabkan ratusan baris token; nilai model sama dengan R02 |
| R03-HTML | Input `<img src=x onerror=alert(1)>` atau `<script>alert(1)</script>` | HTML input tidak aktif di hasil; periksa Chrome Console |
| R03-C06 | 70 kata `hello` | Masih menampilkan truncation 60 token |
| R03-REG | Pesan promosi dan percakapan yang sama dengan R02, ketiga mode | Angka probabilitas sama pada presisi UI (dua desimal) |
| R03-RACE | Klik analisis, segera ganti teks/model saat proses berlangsung | Tidak ada respons lama kembali sebagai hasil aktif |
| R03-MOBILE | 390px, 768px, zoom125% | Kartu dan teks tetap terbaca |

Catat kegagalan apa pun sebagai FAIL; **jangan lakukan UI Freeze sebelum R03-RACE lolos**.

## Backup dan rollback

Installer membuat backup lengkap sebelum mengubah `app.py` dan CSS di:
`outputs\reports\r03_backups\YYYYMMDD_HHMMSS`.

Untuk mengembalikan file setelah server dihentikan:

```powershell
.\.venv\Scripts\python.exe tools\apply_r03.py --project . --rollback
.\.venv\Scripts\python.exe scripts\13_verify_app.py --check
```

Jangan gunakan `git reset --hard` atau `git clean -fd` untuk membatalkan perubahan ini. Installer tidak melakukan commit ataupun push GitHub.

## Bukti sebelum commit

Simpan hasil `R03_PREFLIGHT_PASS`, output unit test, static/smoke PASS, screenshot D01–D03, Attention pada C05, dan screenshot regresi perbandingan model. Setelah seluruh kasus wajib PASS, baru buat commit di branch `uas-development`.
