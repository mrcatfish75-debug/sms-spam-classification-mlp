
# Klasifikasi SMS Spam Menggunakan Deep Learning
### Perbandingan TF-IDF + MLP dan Word2Vec + GRU + Attention

Proyek kelompok mata kuliah **Pengantar Algoritma Deep Learning**, Jalur B.

Repository ini berisi pengembangan sistem klasifikasi SMS Spam dan Ham yang dikerjakan dalam dua tahap:

- **UTS:** Klasifikasi menggunakan TF-IDF dan Multilayer Perceptron (MLP).
- **UAS:** Pengembangan menggunakan Word2Vec, Gated Recurrent Unit (GRU), dan mekanisme Additive Attention.

Penelitian mencakup persiapan dataset, pelatihan model, hyperparameter tuning, studi ablasi, analisis kesalahan, evaluasi final pada data test, serta pengembangan aplikasi demonstrasi berbasis Gradio.

**Status:** Pengembangan model dan evaluasi final selesai. Aplikasi Gradio telah memasuki UI Refinement R03 dengan pengujian regresi akhir yang masih perlu diselesaikan.

---

## 1. Deskripsi Proyek

SMS Spam merupakan pesan tidak diinginkan yang dapat berisi promosi, penawaran mencurigakan, atau upaya penipuan. Klasifikasi otomatis bertujuan membantu membedakan pesan Spam dari pesan normal (Ham) berdasarkan pola teks.

Proyek ini menggunakan dua pendekatan Deep Learning:

**Model UTS — TF-IDF + MLP**

Pesan diubah menjadi representasi numerik menggunakan TF-IDF, kemudian diklasifikasikan dengan Multilayer Perceptron.

**Model UAS — Word2Vec + GRU + Attention**

Pesan direpresentasikan sebagai urutan token dengan embedding Word2Vec, kemudian diproses menggunakan GRU dan Additive Attention.

Pengembangan UAS bertujuan mengeksplorasi pendekatan pemodelan urutan teks serta mekanisme Attention, bukan mengasumsikan bahwa model yang lebih kompleks pasti menghasilkan kinerja lebih tinggi.

Kedua model dievaluasi menggunakan pembagian dataset yang sama agar perbandingan dapat dilakukan secara konsisten.

---

## 2. Tujuan Penelitian

Penelitian memiliki tujuan berikut:

1. Membangun baseline klasifikasi SMS menggunakan TF-IDF dan MLP.
2. Mengembangkan model klasifikasi berbasis Word2Vec, GRU, dan Additive Attention.
3. Melakukan hyperparameter tuning dalam ruang pencarian terbatas.
4. Melaksanakan studi ablasi untuk mengevaluasi kontribusi komponen model.
5. Mengevaluasi dan membandingkan performa model pada data test yang belum digunakan selama pengembangan.
6. Menganalisis kesalahan klasifikasi dan bobot Attention.
7. Mengembangkan aplikasi demonstrasi interaktif untuk pengguna umum dan kebutuhan presentasi akademik.

---

## 3. Dataset

Dataset yang digunakan adalah **SMS Spam Collection** dari UCI Machine Learning Repository.

**Sumber:**

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

**Lisensi:** CC BY 4.0.

### 3.1 Karakteristik Dataset

| Keterangan | Jumlah |
|---|---:|
| Data awal | 5.574 SMS |
| Duplikat dihapus | 415 |
| Data akhir | 5.159 SMS |
| Ham | 4.517 SMS |
| Spam | 642 SMS |

Dataset terdiri atas dua kelas:

- **Ham (0):** Pesan normal atau bukan spam.
- **Spam (1):** Pesan yang diberi label spam.

### 3.2 Pembagian Data

Pembagian dilakukan menggunakan stratified split dengan random seed 42.

| Subset | Jumlah |
|---|---:|
| Training | 4.127 |
| Validation | 516 |
| Test | 516 |
| **Total** | **5.159** |

Data training digunakan untuk membangun model. Data validation digunakan untuk memantau pelatihan, pemilihan konfigurasi, dan pengembangan model.

Data test disimpan terpisah dan baru digunakan untuk evaluasi final setelah protokol penelitian ditetapkan.

**Pencegahan data leakage:**

- Penghapusan duplikat dilakukan sebelum pembagian dataset.
- TF-IDF dipelajari menggunakan data training.
- Word2Vec dan kosakata dibangun dari data training.
- Data validation tidak digunakan untuk melatih embedding.
- Data test tidak digunakan untuk hyperparameter tuning atau pemilihan model.

Dataset mentah, hasil pemrosesan tertentu, dan file split tidak disertakan langsung dalam repository publik.

---

## 4. Metode UTS — TF-IDF + MLP

Pada tahap UTS, sistem dikembangkan menggunakan pendekatan representasi teks TF-IDF dan klasifikasi MLP.

### 4.1 Representasi Teks

Konfigurasi TF-IDF:

- Maksimal 2.000 fitur.
- N-gram: unigram dan bigram.
- Fitting hanya dilakukan menggunakan data training.

### 4.2 Arsitektur MLP

Arsitektur model:

```text
Input TF-IDF (2.000 fitur)
        |
Dense(64, ReLU)
        |
Dropout(0.2)
        |
Dense(32, ReLU)
        |
Dense(1, Sigmoid)
        |
Probabilitas Spam
```

Konfigurasi pelatihan:

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Binary Cross-Entropy |
| Batch size | 32 |
| Maksimum epoch | 15 |
| Early Stopping | Berdasarkan validation loss |
| Patience | 3 |
| Restore best weights | True |
| Threshold klasifikasi | 0.50 |

### 4.3 Hasil Validation MLP

| Metrik | Hasil |
|---|---:|
| Accuracy | 99,03% |
| Precision Spam | 95,38% |
| Recall Spam | 96,88% |
| F1-score Spam | 96,12% |
| Average Precision | 98,89% |

Confusion matrix validation:

```text
[[449,   3],
 [  2,  62]]
```

Hasil pada bagian ini berasal dari validation set, bukan final test.

---

## 5. Metode UAS — Word2Vec + GRU + Attention

Pengembangan UAS menggunakan pendekatan pemodelan urutan teks.

### 5.1 Tokenisasi dan Word2Vec

Representasi teks dibuat dari data training menggunakan Word2Vec.

Konfigurasi utama:

| Komponen | Nilai |
|---|---|
| Word embedding | Word2Vec |
| Embedding dimension | 64 |
| Vocabulary size | 3.443 |
| Maximum sequence length | 60 token |
| PAD token ID | 0 |
| UNK token ID | 1 |
| Random seed | 42 |

Kosakata dan embedding dibentuk tanpa menggunakan data validation maupun test.

Pesan dengan panjang melebihi batas sequence diproses mengikuti konfigurasi pemotongan yang telah ditetapkan.

Token yang tidak ditemukan dalam kosakata ditangani menggunakan mekanisme out-of-vocabulary (OOV).

### 5.2 Arsitektur Model

Model pengembangan menggunakan komponen utama:

```text
Input SMS
    |
Tokenisasi
    |
Sequence Encoding
    |
Word2Vec Embedding
    |
GRU
    |
Additive Attention
    |
Representasi Konteks
    |
Dense + Dropout
    |
Sigmoid
    |
Probabilitas Spam
```

GRU digunakan untuk mempelajari representasi urutan token, sedangkan mekanisme Attention memberikan pembobotan relatif terhadap representasi token.

Bobot Attention digunakan sebagai informasi interpretasi internal model.

**Bobot Attention tidak dianggap sebagai bukti kausal bahwa token tertentu menyebabkan keputusan klasifikasi.**

---

## 6. Hyperparameter Tuning

Tuning dilakukan menggunakan fixed small grid pada data training dan validation.

Empat konfigurasi diuji:

| Trial | GRU Units | Dropout |
|---|---:|---:|
| T1 | 32 | 0.2 |
| T2 | 32 | 0.4 |
| T3 | 64 | 0.2 |
| T4 | 64 | 0.4 |

Pemilihan menggunakan urutan kriteria:

1. F1-score Spam validation tertinggi.
2. Recall Spam validation tertinggi.
3. Average Precision validation tertinggi.
4. Jumlah parameter lebih kecil.
5. Urutan ID trial sebagai tie-break terakhir.

### 6.1 Konfigurasi Terpilih

Konfigurasi yang terpilih adalah **T3**.

| Parameter | Nilai |
|---|---|
| GRU units | 64 |
| Dropout | 0.2 |
| Embedding dimension | 64 |
| Maximum sequence length | 60 |
| Parameters total | 246.433 |
| Maximum epochs | 15 |
| Early Stopping patience | 3 |
| Best epoch berdasarkan val_loss | 3 |
| Threshold | 0.50 |

### 6.2 Hasil Validation T3

| Metrik | Hasil |
|---|---:|
| Accuracy | 99,61% |
| Precision Spam | 100,00% |
| Recall Spam | 96,88% |
| F1-score Spam | 98,41% |
| Average Precision | 98,79% |

Pemilihan T3 dilakukan menggunakan data validation.

Perbedaan metrik validation yang kecil tidak otomatis membuktikan peningkatan generalisasi terhadap data baru.

---

## 7. Studi Ablasi

Studi ablasi dilakukan untuk mengevaluasi kontribusi komponen utama pada model UAS.

Eksperimen mencakup:

- **Model utama:** Word2Vec + GRU + Attention T3.
- **Ablasi Attention:** Model tanpa mekanisme Attention.
- **Ablasi Word2Vec:** Model dengan representasi embedding alternatif tanpa pretrained Word2Vec dari pipeline penelitian.

Ablasi dilakukan menggunakan protokol training dan validation yang konsisten.

Secara umum, model utama mencapai F1-score Spam validation yang lebih tinggi daripada dua konfigurasi ablasi yang diuji.

Namun, beberapa konfigurasi ablasi memperoleh Average Precision yang sedikit lebih tinggi. Karena itu, hasil ablasi tidak dapat ditafsirkan sebagai keunggulan model utama pada seluruh metrik.

Studi ini menunjukkan bahwa kontribusi komponen model perlu dilihat berdasarkan metrik yang digunakan, bukan hanya kompleksitas arsitektur.

---

## 8. Analisis Kesalahan dan Attention

Analisis kesalahan dilakukan terhadap prediksi model pada validation set.

Untuk model UAS, dilakukan pemeriksaan pada sampel salah klasifikasi serta contoh tambahan yang berada dekat ambang keputusan.

Model T3 menghasilkan dua false negative pada validation set.

Visualisasi Attention memperlihatkan lima token dengan bobot tertinggi dan nilai numeriknya.

Informasi yang dianalisis mencakup:

- Token dengan bobot Attention tertinggi.
- Distribusi bobot relatif.
- Token yang tidak ditemukan dalam kosakata.
- Perbedaan keluaran model MLP dan GRU.
- Keterbatasan interpretasi hasil model.

Interpretasi dilakukan secara hati-hati karena bobot Attention tidak secara langsung membuktikan penyebab suatu prediksi.

---

## 9. Evaluasi Final Test

Evaluasi final dilakukan setelah konfigurasi model, threshold, dan protokol evaluasi dibekukan.

**Dataset test hanya dievaluasi satu kali.**

Jumlah data test:

- Total: 516 SMS.
- Ham: 452 SMS.
- Spam: 64 SMS.

### 9.1 Perbandingan Hasil Final

| Metrik | MLP Baseline (UTS) | GRU + Attention T3 (UAS) |
|---|---:|---:|
| Accuracy | 98,64% | 98,45% |
| Precision Spam | 95,24% | 96,67% |
| Recall Spam | 93,75% | 90,63% |
| F1-score Spam | 94,49% | 93,55% |
| Average Precision | 98,13% | 98,95% |

### 9.2 Confusion Matrix MLP

```text
[[449,   3],
 [  4,  60]]
```

### 9.3 Confusion Matrix GRU T3

```text
[[450,   2],
 [  6,  58]]
```

### 9.4 Interpretasi Hasil

Hasil final menunjukkan adanya trade-off antara kedua model.

**MLP baseline** memiliki:

- Accuracy lebih tinggi.
- Recall Spam lebih tinggi.
- F1-score Spam lebih tinggi.

**GRU + Attention T3** memiliki:

- Precision Spam lebih tinggi.
- Average Precision lebih tinggi.
- False positive lebih sedikit.

Perbedaan performa menunjukkan bahwa model yang lebih kompleks tidak selalu memberikan peningkatan pada semua metrik.

Tidak ada klaim bahwa GRU + Attention lebih unggul secara universal daripada MLP.

**Catatan:** Average Precision (AP) dihitung sebagai metrik evaluasi precision-recall. Dalam proyek ini, AP tidak boleh disamakan dengan luas kurva precision-recall berdasarkan integrasi trapezoidal.

---

## 10. Aplikasi Demonstrasi Gradio

Sebagai bagian dari pengembangan UAS, dibuat aplikasi web lokal menggunakan **Gradio 6**.

Aplikasi dirancang sebagai sarana demonstrasi model untuk dosen, mahasiswa Teknik Informatika, dan pengguna umum.

### 10.1 Fitur Aplikasi

Fitur utama:

1. Memasukkan satu SMS berbahasa Inggris.
2. Menggunakan contoh pesan biasa atau promosi.
3. Memilih model GRU + Attention (UAS).
4. Memilih model MLP baseline (UTS).
5. Membandingkan keluaran kedua model.
6. Menampilkan label Spam atau Ham.
7. Menampilkan perkiraan probabilitas Spam.
8. Menampilkan ambang keputusan 50%.
9. Menampilkan lima token dengan bobot Attention tertinggi untuk GRU.
10. Menampilkan diagnostik token dan OOV.
11. Memberikan penjelasan interpretasi dan keterbatasan model.

Aplikasi tidak melakukan pelatihan ulang ketika pengguna menjalankan prediksi.

### 10.2 Desain Antarmuka

Pengembangan antarmuka menggunakan pendekatan **Human-Centered ML Workbench**.

Prinsip desain:

- Alur sederhana: masukkan pesan, pilih model, baca hasil.
- Informasi utama lebih mudah dipahami pengguna awam.
- Rincian teknis tersedia untuk mahasiswa dan dosen.
- Konsistensi warna, kontras, dan tipografi.
- Tampilan responsif untuk desktop dan layar sempit.
- Penjelasan probabilitas yang tidak menyesatkan.

### 10.3 UI Refinement R02

Refinement R02 memperbaiki:

- Kontras hasil.
- Hierarki informasi klasifikasi.
- Label Spam/Ham.
- Penyajian probabilitas.
- Tampilan perbandingan dua model.
- Visualisasi Attention.
- Responsivitas dan aksesibilitas dasar.

Pengujian R02 meliputi tujuh kelompok:

- A: Klasifikasi model.
- B: Attention.
- C: Validasi dan ketahanan input.
- D: Konsistensi hasil.
- E: Responsivitas.
- F: Pengoperasian keyboard.
- G: Stabilitas aplikasi.

Pengujian menunjukkan bahwa fitur utama berjalan, tetapi masih ditemukan masalah pada penampilan hasil lama ketika input atau mode berubah dan visualisasi token yang sangat panjang.

### 10.4 UI Refinement R03

R03 dikembangkan untuk menangani temuan pengujian R02.

Ruang lingkup perbaikan:

- Invalidasi hasil ketika teks SMS berubah.
- Invalidasi hasil ketika model atau mode analisis berubah.
- Penanganan tampilan token Attention yang sangat panjang.
- Pengamanan penyajian HTML dinamis.
- Penyempurnaan status hasil yang memerlukan analisis ulang.

R03 mempertahankan model, tokenizer, threshold, dan hasil evaluasi yang sebelumnya sudah dibekukan.

Pengujian otomatis R03 yang telah dijalankan mencakup:

- Pemeriksaan sintaks Python.
- Pemeriksaan statis R03.
- Dua belas unit test.
- Verifikasi SHA-256 artefak model.
- Verifikasi tokenizer dan threshold.
- Synthetic smoke test.

Seluruh pemeriksaan otomatis tersebut telah memperoleh hasil PASS pada lingkungan lokal pengembangan.

**Status R03:** Perbaikan telah diterapkan secara lokal. Pengujian regresi interaksi akhir tetap diperlukan sebelum antarmuka dinyatakan UI Freeze.

---

## 11. Struktur Repository

Struktur utama proyek:

```text
sms-spam-classification-mlp/
|
|-- app.py
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- splits/
|
|-- docs/
|
|-- models/
|
|-- notebooks/
|
|-- outputs/
|   |-- figures/
|   `-- reports/
|
|-- payload/
|
|-- scripts/
|
|-- src/
|   |-- __init__.py
|   `-- uas_inference.py
|
|-- tests/
|
|-- tools/
|
|-- ui/
|   |-- uas_theme.css
|   `-- r03_helpers.py
|
|-- README.md
|
|-- requirements-hari1.txt
|-- requirements-tahap2.txt
|-- requirements-uas-tahap1.txt
|-- requirements-uas-tahap5.txt
|-- requirements-uas-tahap6.txt
|
|-- PANDUAN_UAS_TAHAP6.md
|-- PANDUAN_UAS_UI_R02.md
`-- PANDUAN_R03.md
```

Sebagian direktori menyimpan artefak lokal yang tidak otomatis disertakan dalam repository publik.

---

## 12. Instalasi dan Menjalankan Aplikasi

### 12.1 Persyaratan

Lingkungan pengembangan yang digunakan:

- Windows.
- Python 3.13.
- TensorFlow 2.21.
- Gradio 6.
- CPU.
- RAM 8 GB.

GPU tidak diwajibkan untuk menjalankan aplikasi pada lingkungan pengembangan ini.

### 12.2 Clone Repository

```powershell
git clone https://github.com/mrcatfish75-debug/sms-spam-classification-mlp.git

cd sms-spam-classification-mlp
```

Gunakan branch pengembangan jika pembaruan UAS belum digabungkan ke branch utama:

```powershell
git switch uas-development
```

### 12.3 Buat Virtual Environment

```powershell
py -3.13 -m venv .venv
```

### 12.4 Instal Dependensi

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install -r requirements-uas-tahap6.txt
```

### 12.5 Siapkan Artefak Model

Aplikasi memerlukan file model terlatih serta artefak preprocessing yang sesuai.

Model GRU terpilih menggunakan:

```text
models/gru_attention_tuned.keras
models/gru_attention_tuned_reader.keras
```

Model MLP dan artefak TF-IDF juga harus tersedia sesuai path yang digunakan oleh modul inferensi.

**Penting:** File model, vectorizer, dan artefak pemrosesan tertentu tidak otomatis diunggah ke repository publik karena pengaturan `.gitignore`.

Untuk menjalankan aplikasi dari hasil clone baru, file yang diperlukan harus disediakan melalui paket artefak model yang sesuai.

Tautan distribusi model akan dicantumkan setelah paket artefak final disiapkan dan diverifikasi.

### 12.6 Verifikasi Integritas

Sebelum menjalankan aplikasi:

```powershell
.\.venv\Scripts\python.exe scripts\13_verify_app.py --check
```

Target:

```text
STATUS: UAS_STAGE6_STATIC_VERIFIED
```

Kemudian jalankan synthetic smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\13_verify_app.py --smoke
```

Target:

```text
STATUS: UAS_STAGE6_SMOKE_PASS
```

### 12.7 Jalankan Aplikasi

```powershell
.\.venv\Scripts\python.exe app.py
```

Buka browser:

http://127.0.0.1:7860

Alamat tersebut merupakan server lokal dan hanya dapat digunakan ketika aplikasi sedang berjalan pada komputer yang bersangkutan.

---

## 13. Pengujian R03

Pengujian regresi diperlukan untuk memastikan perubahan antarmuka tidak memengaruhi keluaran model.

### 13.1 Pemeriksaan Otomatis

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py

.\.venv\Scripts\python.exe tools\r03_verify_local.py --project .

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_r03_*.py" -v

.\.venv\Scripts\python.exe scripts\13_verify_app.py --check

.\.venv\Scripts\python.exe scripts\13_verify_app.py --smoke
```

### 13.2 Pemeriksaan Manual

Pengujian manual meliputi:

- Pergantian input tanpa analisis ulang.
- Pergantian model dan mode perbandingan.
- Konsistensi probabilitas.
- Aktivasi dan penonaktifan Attention.
- Pengujian input ekstrem.
- Pengujian markup HTML.
- Tampilan layar sempit.
- Navigasi keyboard.
- Stabilitas setelah refresh.
- Pemeriksaan terminal dan browser Console.

Pengujian manual R03 harus diselesaikan sebelum status UI Final diberikan.

---

## 14. Keterbatasan Penelitian

Proyek ini memiliki sejumlah keterbatasan.

**1. Dataset historis**

Data berasal dari SMS berbahasa Inggris yang dikumpulkan pada periode sebelumnya. Distribusinya tidak selalu mewakili karakteristik SMS modern.

**2. Bahasa**

Model tidak dilatih secara khusus untuk bahasa Indonesia atau berbagai bahasa lain.

**3. Generalisasi**

Kinerja pada dataset test tidak menjamin hasil yang sama pada pesan baru yang berbeda dari distribusi pelatihan.

**4. Probabilitas prediksi**

Angka probabilitas spam merupakan skor keluaran model yang belum diuji sebagai probabilitas terkalibrasi.

**5. Interpretasi Attention**

Bobot Attention hanya menggambarkan pembobotan internal yang dihasilkan model. Bobot tersebut bukan bukti hubungan sebab-akibat.

**6. Ketahanan terhadap perubahan teks**

Perubahan format atau simbol dapat menghasilkan perbedaan prediksi yang besar, terutama pada input di luar karakteristik data pelatihan.

**7. Kegunaan aplikasi**

Aplikasi merupakan prototipe akademik, bukan produk keamanan komersial atau sistem penyaringan SMS produksi.

---

## 15. Etika dan Privasi

Pengembangan memperhatikan penggunaan data dan interpretasi hasil secara bertanggung jawab.

Prinsip yang diterapkan:

- Tidak menggunakan data test untuk tuning.
- Tidak mengklaim probabilitas sebagai jaminan keamanan.
- Tidak menyatakan Attention sebagai penjelasan kausal.
- Tidak memasukkan data SMS pribadi atau rahasia ke repository publik.
- Tidak menjadikan prediksi model sebagai satu-satunya dasar keputusan keamanan.
- Memisahkan artefak model, dataset, dan laporan internal dari source code yang dipublikasikan.

Pengguna tidak disarankan memasukkan OTP, password, nomor identitas, maupun informasi pribadi ke aplikasi yang di-host secara publik.

---

## 16. Status Pengembangan

| Tahap | Status |
|---|---|
| UTS — TF-IDF + MLP | Selesai |
| UAS 1 — Persiapan sequence dan Word2Vec | Selesai |
| UAS 2 — Pelatihan GRU + Attention | Selesai |
| UAS 3 — Hyperparameter tuning | Selesai |
| UAS 4 — Studi ablasi dan Attention | Selesai |
| UAS 5 — Protokol dan final test | Selesai |
| UAS 6 — Aplikasi Gradio | Berfungsi |
| UI Refinement R02 | Pengujian selesai dengan temuan |
| UI Refinement R03 | Diterapkan secara lokal; regresi final perlu diselesaikan |
| Deployment publik | Belum dikonfirmasi |

Hasil final test tidak diubah oleh proses refinement antarmuka.

---

## 17. Repository dan Referensi Dataset

**Repository GitHub:**

https://github.com/mrcatfish75-debug/sms-spam-classification-mlp

**Dataset:**

Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering*.

UCI SMS Spam Collection:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## Catatan Akhir

Repository ini dibuat untuk keperluan pembelajaran, penelitian, dan demonstrasi akademik.

Pengembangan UTS dan UAS menggunakan dataset serta protokol penelitian yang terkontrol. Hasil menunjukkan bahwa perbedaan arsitektur menghasilkan trade-off metrik, bukan peningkatan yang selalu konsisten.

Aplikasi Gradio digunakan untuk memperlihatkan proses inferensi secara interaktif, bukan sebagai sistem keamanan SMS yang telah divalidasi untuk penggunaan produksi.
