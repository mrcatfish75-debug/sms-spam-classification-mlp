# UAS Tahap 1 - Word2Vec dan Tokenisasi SMS

Paket ini melanjutkan UTS di branch `uas-development`. Tidak mengubah script UTS, hasil
validasi MLP, pembagian train/validation/test, atau file test.

## Isi paket

- `scripts/06_prepare_sequences.py` - preprocessing sekuensial dan Word2Vec.
- `requirements-uas-tahap1.txt` - tambahan dependency untuk tahap ini.

## Cara memasang (Windows PowerShell 5.1)

1. Pastikan root proyek adalah `D:\1. PROJEk DL\SMS-Spam-Indonesia`.
2. Pastikan `git branch --show-current` menampilkan `uas-development`.
3. Ekstrak paket ke root proyek dengan **Extract All**. Pilih `Copy and Replace`
   hanya bila telah memeriksa apakah berkas yang sama pernah Anda buat.
   Alternatif perintah ekstrak yang jelas tercantum di panduan chat.
4. Periksa file tersedia:

   ```powershell
   Test-Path 'scripts\06_prepare_sequences.py'
   Test-Path 'requirements-uas-tahap1.txt'
   ```

5. Instal satu dependency baru pada venv Python 3.13 yang sudah dipakai UTS:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap1.txt
   .\.venv\Scripts\python.exe -c "import gensim; print('gensim:', gensim.__version__)"
   ```

6. Jalankan:

   ```powershell
   .\.venv\Scripts\python.exe scripts\06_prepare_sequences.py
   ```

7. Periksa hasil:

   ```powershell
   $m = Get-Content 'outputs\reports\uas_sequence_word2vec_manifest.json' -Raw | ConvertFrom-Json
   $m.status
   $m.scope
   $m.sequence_train_shape
   $m.sequence_validation_shape
   $m.embedding_shape
   $m.test_status
   git status --short
   ```

## File keluaran lokal (tidak diunggah ke GitHub)

- `data/processed/sequences_train.npz`
- `data/processed/sequences_validation.npz`
- `models/sequence_vocabulary.json`
- `models/sequence_tokenizer_config.json`
- `models/word2vec_embedding_matrix.npy`
- `models/word2vec_train_only.kv`
- `outputs/reports/uas_sequence_word2vec_manifest.json`

Hanya **train** dipakai untuk membangun vocabulary dan melatih Word2Vec. Validation
dipakai untuk pengukuran cakupan dan penyiapan input; test **tidak dibaca**.

Catatan penelitian: Word2Vec pada paket ini **dilatih dari nol** dengan 4.127 SMS
training, bukan hasil download embedding pralatih. Hal ini akan dituliskan
secara jujur dalam laporan UAS. Pembelajaran lanjutan embedding dilakukan saat
model GRU dilatih pada tahap berikutnya.

Jika file telah benar dan hasil PASS, commit script dan requirements pada branch
`uas-development` setelah Anda dan anggota kelompok memeriksa kodenya.

