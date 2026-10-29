"""Analyze misclassified SMS on VALIDATION only, without touching the held-out TEST.

Run at project root (Windows PowerShell):
    .\\.venv\\Scripts\\python.exe scripts\\05_analyze_validation_errors.py

Prerequisites: 03_split_tfidf.py and 04_train_mlp.py completed successfully.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "outputs" / "reports"


def main() -> None:
    split_manifest_path = REPORTS / "split_tfidf_manifest.json"
    results_path = REPORTS / "hasil_validasi_mlp.json"
    validation_path = SPLITS / "sms_validation.csv"
    vectorizer_path = MODELS / "tfidf_vectorizer.joblib"
    model_path = MODELS / "mlp_baseline.keras"

    required_files = [split_manifest_path, results_path, validation_path, vectorizer_path, model_path]
    missing = [str(p.relative_to(ROOT)) for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError("File tahap sebelumnya belum tersedia: " + ", ".join(missing))

    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "SPLIT_TFIDF_PASS":
        raise RuntimeError("Manifest split belum PASS; jangan lanjutkan.")
    if results.get("status") != "MLP_TRAINED_VALIDATION_ONLY":
        raise RuntimeError("Training baseline belum menghasilkan status valid.")
    if results.get("test_status") != "LOCKED_NOT_EVALUATED":
        raise RuntimeError("Status TEST tidak sesuai protokol LOCKED.")

    validation = pd.read_csv(validation_path, keep_default_na=False)
    needed = {"source_row", "label", "text"}
    if not needed.issubset(validation.columns):
        raise ValueError(f"Kolom validation wajib: {sorted(needed)}")
    if len(validation) != manifest["validation"]["n"]:
        raise RuntimeError("Jumlah baris validation berbeda dari manifest.")
    if not validation["label"].isin(["ham", "spam"]).all():
        raise RuntimeError("Label validation tidak valid.")

    vectorizer = joblib.load(vectorizer_path)  # Hanya file yang dibuat sendiri, bukan file dari pihak tidak tepercaya.
    x_val = vectorizer.transform(validation["text"]).astype(np.float32).toarray()
    if x_val.shape[1] != manifest["tfidf"]["n_features"]:
        raise RuntimeError("Jumlah fitur TF-IDF tidak sama dengan manifest.")

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow belum terpasang dalam .venv proyek.") from exc
    model = tf.keras.models.load_model(model_path, compile=False)
    score = model.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    threshold = float(results.get("threshold", 0.5))
    pred = (score >= threshold).astype(int)
    truth = (validation["label"].to_numpy() == "spam").astype(int)
    cm = confusion_matrix(truth, pred, labels=[0, 1]).tolist()

    expected_cm = results.get("confusion_matrix_validation_labels_0ham_1spam")
    matches_reference = cm == expected_cm
    if not matches_reference:
        raise RuntimeError(
            "Confusion matrix tidak sama dengan laporan baseline. "
            f"Sekarang: {cm}; laporan: {expected_cm}. "
            "Periksa apakah model, vectorizer, dan split tidak berubah. Tidak menulis hasil."
        )

    errors = validation.loc[pred != truth, ["source_row", "label", "text"]].copy()
    mismatch_indexes = np.flatnonzero(pred != truth)
    errors["predicted_label"] = ["spam" if pred[i] else "ham" for i in mismatch_indexes]
    errors["spam_probability"] = np.round(score[mismatch_indexes], 6)
    errors["error_type"] = [
        "false_positive" if truth[i] == 0 else "false_negative" for i in mismatch_indexes
    ]
    errors["label_needs_review"] = ""
    errors["possible_cause"] = ""
    errors["reviewer_notes"] = ""
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS / "validation_error_cases.csv"
    errors.to_csv(out_csv, index=False, encoding="utf-8-sig")

    metrics = {
        "accuracy": round(float(accuracy_score(truth, pred)), 6),
        "precision_spam": round(float(precision_score(truth, pred, zero_division=0)), 6),
        "recall_spam": round(float(recall_score(truth, pred, zero_division=0)), 6),
        "f1_spam": round(float(f1_score(truth, pred, zero_division=0)), 6),
    }
    summary = {
        "status": "VALIDATION_ERRORS_ANALYZED_AUTOMATICALLY",
        "scope": "VALIDATION_ONLY__TEST_NOT_READ",
        "validation_count": int(len(validation)),
        "errors_count": int(len(errors)),
        "false_positives": int(cm[0][1]),
        "false_negatives": int(cm[1][0]),
        "confusion_matrix_actual_rows_ham_spam_predicted_cols_ham_spam": cm,
        "same_confusion_matrix_as_baseline": matches_reference,
        "metrics_recomputed": metrics,
        "manual_error_reason_review": "PENDING",
        "note": "CSV menampilkan SMS asli (mungkin ada nomor telepon); jangan publikasi tanpa penyamaran data pribadi.",
    }
    out_json = REPORTS / "validation_error_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 66)
    print("TAHAP 3A — AUDIT SALAH PREDIKSI PADA VALIDATION SAJA")
    print("=" * 66)
    print("Validation rows:", len(validation))
    print("Confusion matrix sesuai baseline:", matches_reference)
    print("False positive:", cm[0][1], "| False negative:", cm[1][0])
    print("Total prediksi salah:", len(errors))
    print("Output CSV:", out_csv.relative_to(ROOT))
    print("Output JSON:", out_json.relative_to(ROOT))
    print("TEST TIDAK DIBACA. Jangan jalankan evaluasi TEST pada tahap UTS ini.")


if __name__ == "__main__":
    main()
