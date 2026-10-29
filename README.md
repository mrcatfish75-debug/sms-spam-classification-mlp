# Proyek UTS — Klasifikasi SMS Spam dan Non-Spam Bahasa Indonesia (Topik Mandiri B)

**Model UTS**: TF-IDF + Multilayer Perceptron (MLP).  
**Model UAS (rencana)**: LSTM/GRU dan demo web Gradio.

Mulai dari **[README_HARI1.md](README_HARI1.md)**. Pada paket awal, baru tersedia skrip audit dataset. Model dan hasil eksperimen belum dibuat, sehingga belum ada nilai metrik performa yang diklaim.

## Struktur

```text
data/raw/               dataset asli, tidak disertakan di repo publik

data/processed/         hasil audit, tidak disertakan di repo publik
notebooks/              notebook eksperimen (minggu berikutnya)
scripts/01_audit_dataset.py
src/                    modul preprocessing dan training (tahap selanjutnya)
models/                 model terlatih (tahap selanjutnya)
outputs/reports/         laporan audit lokal
outputs/figures/         grafik EDA lokal
requirements-hari1.txt
README_HARI1.md
```

Dataset: https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset (5.572 baris menurut kartu dataset, kolom `text_id` hasil terjemahan mesin, lisensi tercantum `gpl`). Hasil akhir harus tetap menjalani audit data.
