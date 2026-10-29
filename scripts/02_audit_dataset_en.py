"""Audit UCI SMS Spam Collection (English originals) — run from project root.

Required input: data/raw/SMSSpamCollection (tab-delimited file, no extension).
Optional input: ZIP in data/raw/, if not already extracted.
Outputs use suffix _en to avoid overwriting Indonesian audit artifacts.
Python 3.13 + pandas + matplotlib.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW = RAW_DIR / "SMSSpamCollection"
CLEAN_PATH = ROOT / "data" / "processed" / "sms_en_clean.csv"
REPORT_DIR = ROOT / "outputs" / "reports"
FIGURE_DIR = ROOT / "outputs" / "figures"
MIN_SAMPLES = 5000
SEED = 42
SOURCE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"


def normalize_duplicate_key(text: str) -> str:
    # Conservative: preserve punctuation, digits, URLs and phone numbers.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def find_or_extract_input() -> Path:
    if RAW.is_file():
        return RAW
    candidates = sorted(RAW_DIR.glob("*.zip"))
    for archive_path in candidates:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                matches = [name for name in archive.namelist() if Path(name).name == "SMSSpamCollection"]
                if not matches:
                    continue
                data = archive.read(matches[0])
                RAW.parent.mkdir(parents=True, exist_ok=True)
                RAW.write_bytes(data)
                print(f"Input UCI diekstrak dari {archive_path.name} -> {RAW.name}")
                return RAW
        except zipfile.BadZipFile:
            print(f"Peringatan: {archive_path.name} bukan ZIP valid; dilewati.")
    raise SystemExit(
        "DATASET ASLI UCI BELUM ADA. Unduh ZIP dari https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
        "Ekstrak 'SMSSpamCollection' ke data/raw/ atau letakkan ZIP langsung di data/raw/.\n"
        "Jangan memakai 'data-augmented.csv' dari dataset multibahasa untuk langkah ini."
    )


def load_file(path: Path) -> tuple[pd.DataFrame, str]:
    blob = path.read_bytes()
    text, encoding = None, None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = blob.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise SystemExit("Gagal mendekode dataset menggunakan UTF-8/Windows-1252/Latin-1.")
    rows = []
    invalid_format = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if "\t" not in line:
            invalid_format.append(line_no)
            continue
        label, sms = line.split("\t", 1)
        rows.append({"source_row": line_no, "label": label.strip().lower(), "text": sms.strip()})
    if invalid_format:
        raise SystemExit(
            f"Ada {len(invalid_format)} baris tanpa pemisah tab (misalnya baris {invalid_format[:5]}). "
            "Periksa file sumber; audit dibatalkan untuk mencegah pembacaan keliru."
        )
    return pd.DataFrame(rows, columns=["source_row", "label", "text"]), encoding


def main() -> None:
    print("=" * 65)
    print("AUDIT DATASET INGGRIS ASLI — UCI SMS SPAM COLLECTION")
    print("=" * 65)
    raw_path = find_or_extract_input()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw, detected_encoding = load_file(raw_path)
    raw_count = int(len(raw))
    null_text = int(raw["text"].isna().sum())
    null_label = int(raw["label"].isna().sum())
    empty_text = int((raw["text"].fillna("").str.strip() == "").sum())
    invalid_label = int((~raw["label"].isin(["ham", "spam"])).sum())
    valid = raw[(raw["text"].str.strip() != "") & raw["label"].isin(["ham", "spam"])].copy()
    valid["key"] = valid["text"].map(normalize_duplicate_key)

    # Avoid letting identical/normalized-identical SMS with divergent labels into the model.
    conflict_flags = valid.groupby("key")["label"].nunique()
    conflict_keys = set(conflict_flags.index[conflict_flags > 1])
    conflicted = valid[valid["key"].isin(conflict_keys)].copy()
    conflict_path = REPORT_DIR / "konflik_label_en.csv"
    if not conflicted.empty:
        conflicted.drop(columns="key").to_csv(conflict_path, index=False, encoding="utf-8-sig")
    elif conflict_path.exists():
        conflict_path.unlink()
    kept = valid[~valid["key"].isin(conflict_keys)].copy()
    dup_removed = int(kept.duplicated("key").sum())
    clean = kept.drop_duplicates("key", keep="first").drop(columns="key").reset_index(drop=True)
    clean["panjang_karakter"] = clean["text"].str.len()
    clean["panjang_kata"] = clean["text"].str.split().str.len()
    clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

    # Stratified manual sample only; not a substitute for full dataset quality validation.
    sampled_parts = []
    for label in ("ham", "spam"):
        subset = clean[clean["label"] == label]
        if len(subset):
            sampled_parts.append(subset.sample(n=min(20, len(subset)), random_state=SEED))
    if sampled_parts:
        sample = pd.concat(sampled_parts).sort_values("source_row").copy()
        sample["teks_terbaca_ya_tidak"] = ""
        sample["label_sesuai_ya_tidak"] = ""
        sample["catatan_reviewer"] = ""
        sample.to_csv(REPORT_DIR / "sample_pemeriksaan_en.csv", index=False, encoding="utf-8-sig")

    counts = {name: int((clean["label"] == name).sum()) for name in ("ham", "spam")}
    ax = pd.Series(counts).plot.bar(rot=0)
    ax.set_title("Distribusi Kelas SMS — UCI English (Setelah Deduplikasi)")
    ax.set_ylabel("Jumlah Pesan")
    ax.set_xlabel("Kelas")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "distribusi_label_en.png", dpi=170)
    plt.close()

    if len(clean):
        plt.hist(clean["panjang_kata"], bins=35)
        plt.title("Distribusi Panjang SMS — UCI English")
        plt.xlabel("Jumlah Kata")
        plt.ylabel("Frekuensi")
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "panjang_pesan_en.png", dpi=170)
        plt.close()

    report = {
        "dataset": "UCI SMS Spam Collection (original English)",
        "dataset_source": SOURCE,
        "dataset_license": "CC BY 4.0 (per halaman resmi UCI)",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "input_encoding": detected_encoding,
        "data_mentah": raw_count,
        "baris_text_null": null_text,
        "baris_label_null": null_label,
        "baris_text_kosong": empty_text,
        "baris_label_invalid": invalid_label,
        "baris_konflik_label_dikarantina": int(len(conflicted)),
        "baris_duplikat_dihapus": dup_removed,
        "data_bersih_unik": int(len(clean)),
        "distribusi_label_bersih": counts,
        "minimum_dosen": MIN_SAMPLES,
        "lolos_ambang_jumlah": bool(len(clean) >= MIN_SAMPLES),
        "status_review_sampel": "PENDING: periksa sample_pemeriksaan_en.csv",
        "status_persetujuan_topik": "PENDING: topik mandiri membutuhkan persetujuan dosen",
        "status_training": "BELUM DIMULAI; lanjut setelah validasi kualitas dan izin topik",
        "catatan": "Jangan fit TF-IDF atau lakukan resampling sebelum membagi train/validation/test. Deduplikasi dilakukan sebelum split.",
    }
    result_path = REPORT_DIR / "audit_dataset_en.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Data mentah: {raw_count:,}")
    print(f"Baris konflik label dikarantina: {len(conflicted):,}")
    print(f"Duplikat dihapus: {dup_removed:,}")
    print(f"Data bersih unik: {len(clean):,}")
    print(f"Distribusi kelas: {counts}")
    print(f"Minimum 5.000 terpenuhi: {report['lolos_ambang_jumlah']}")
    print("Hasil baru:")
    for path in (CLEAN_PATH, result_path, REPORT_DIR / "sample_pemeriksaan_en.csv",
                 FIGURE_DIR / "distribusi_label_en.png", FIGURE_DIR / "panjang_pesan_en.png"):
        print("-", path.relative_to(ROOT))
    if not report["lolos_ambang_jumlah"]:
        print("PERINGATAN: sampel unik < 5000; jangan memaksakan kelayakan.")
    else:
        print("LOLOS JUMLAH — tetap periksa contoh teks dan persetujuan topik.")


if __name__ == "__main__":
    main()
