r"""UAS Tahap 5B: one-time, explicit held-out TEST evaluation of frozen MLP and T3.

SAFE DEFAULT (does not open TEST):
    .\.venv\Scripts\python.exe scripts\12_evaluate_final_test.py --status

ONE-TIME EVALUATION, ONLY after 5A freeze and user approval:
    .\.venv\Scripts\python.exe scripts\12_evaluate_final_test.py --evaluate-test --confirm EVALUATE_LOCKED_TEST_ONCE

Never fits/trains any model. Never changes tokenizer/model/split. A one-time marker
prevents accidental re-evaluation, even after an interrupted execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
PREP_SCRIPT = ROOT / "scripts" / "11_prepare_final_evaluation.py"
FREEZE = REPORTS / "uas_final_protocol_lock.json"
RUN_MARKER = REPORTS / "uas_final_test_once.lock"
FINAL_RESULT = REPORTS / "uas_final_test_results.json"
COMPARISON_CSV = REPORTS / "uas_final_test_comparison.csv"
COMPARISON_PNG = FIGURES / "uas_final_test_model_comparison.png"
CONFUSION_PNG = FIGURES / "uas_final_test_confusion_matrices.png"
THRESHOLD = 0.5


def load_preparation():
    if not PREP_SCRIPT.is_file():
        raise FileNotFoundError("Tahap 5A belum dipasang: scripts/11_prepare_final_evaluation.py")
    spec = importlib.util.spec_from_file_location("_stage5_protocol_verifier", PREP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Tidak bisa memuat pemeriksa Tahap 5A.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_freeze(preparation) -> tuple[dict, dict]:
    lock = preparation.read_json(FREEZE)
    if (lock.get("status") != "UAS_FINAL_EVALUATION_PROTOCOL_FROZEN"
            or lock.get("scope") != "VALIDATION_ONLY__TEST_NOT_READ"
            or lock.get("test_status") != "LOCKED_NOT_EVALUATED"
            or lock.get("threshold") != THRESHOLD
            or lock.get("selected_trial") != "T3"
            or lock.get("model_uts") != "models/mlp_baseline.keras"
            or lock.get("model_uas") != "models/gru_attention_tuned.keras"
            or lock.get("test_expected_count_from_existing_split_manifest") != 516
            or lock.get("split_seed") != 42):
        raise RuntimeError("Freeze protokol tidak sesuai konfigurasi UTS/UAS terkunci.")
    current_hashes = preparation.hashes_for_freeze()
    if lock.get("input_sha256") != current_hashes:
        differences = [key for key in current_hashes
                       if lock.get("input_sha256", {}).get(key) != current_hashes[key]]
        raise RuntimeError("Artefak berubah setelah freeze, STOP: " + ", ".join(differences))
    # Validasi ulang TRAIN/VALIDATION; sama sekali belum membaca data TEST.
    context = preparation.compute_validation_context()
    if (context["cm_mlp"] != lock["validation_confusion_matrix_mlp"]
            or context["cm_gru"] != lock["validation_confusion_matrix_t3"]
            or context["mlp_metrics"] != lock["validation_mlp_metrics"]
            or context["gru_metrics"] != lock["validation_t3_metrics"]):
        raise RuntimeError("Prediksi validation berubah setelah freeze.")
    expected_cases = preparation.expected_five_error_rows(context)
    preparation.validate_five_error_file(expected_cases, require_review=True)
    if [entry["source_row"] for entry in expected_cases] != lock["five_error_case_ids"]:
        raise RuntimeError("Lima kesalahan MLP berbeda dari protokol terkunci.")
    return lock, context


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def read_and_validate_test(preparation, context: dict, lock: dict) -> tuple[pd.DataFrame, str]:
    """Read the bytes of the test CSV exactly once, only after explicit run marker."""
    test_path = SPLITS / "sms_test.csv"
    raw = test_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    test = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", keep_default_na=False)
    required = {"source_row", "label", "text"}
    if not required.issubset(test.columns):
        raise RuntimeError("Kolom test tidak sesuai dengan train/validation.")
    expected = context["split_manifest"]["test_locked"]
    if len(test) != lock["test_expected_count_from_existing_split_manifest"]:
        raise RuntimeError("Jumlah TEST berubah dari manifest.")
    if test["label"].value_counts().to_dict() != expected["class_count"]:
        raise RuntimeError("Distribusi kelas TEST berbeda dari manifest UTS.")
    if (test["source_row"].duplicated().any() or test["text"].str.strip().eq("").any()
            or (~test["label"].isin(["ham", "spam"])).any()):
        raise RuntimeError("Ada ID ganda, teks kosong, atau label invalid pada TEST.")

    train = pd.read_csv(SPLITS / "sms_train.csv", encoding="utf-8-sig", keep_default_na=False)
    val = context["validation"]
    all_ids = [set(df["source_row"]) for df in (train, val, test)]
    all_keys = [set(df["text"].map(canonical)) for df in (train, val, test)]
    if (len(all_keys[2]) != len(test)
            or any(all_ids[i] & all_ids[j] for i, j in ((0, 1), (0, 2), (1, 2)))
            or any(all_keys[i] & all_keys[j] for i, j in ((0, 1), (0, 2), (1, 2)))):
        raise RuntimeError("Potensi leakage / duplikasi antara TRAIN, VALIDATION, dan TEST.")
    if len(train) + len(val) + len(test) != 5159:
        raise RuntimeError("Total baris tidak lagi 5159.")

    # Pastikan test sama persis dengan salah satu baris data bersih menurut ID dan teks.
    clean = pd.read_csv(ROOT / "data/processed/sms_en_clean.csv", encoding="utf-8-sig", keep_default_na=False)
    if (len(clean) != 5159 or clean["source_row"].duplicated().any()
            or set(clean["source_row"]) != (all_ids[0] | all_ids[1] | all_ids[2])):
        raise RuntimeError("Sumber dataset tidak lagi sama dengan gabungan TRAIN/VALIDATION/TEST.")
    by_id = clean.set_index("source_row")[["label", "text"]]
    if (not test["source_row"].isin(by_id.index).all()
            or not np.array_equal(by_id.loc[test["source_row"], "label"].to_numpy(), test["label"].to_numpy())
            or not np.array_equal(by_id.loc[test["source_row"], "text"].to_numpy(), test["text"].to_numpy())):
        raise RuntimeError("Sumber baris TEST tidak sama dengan data bersih UCI yang dikunci.")
    return test, digest


def encode_test(texts: pd.Series) -> np.ndarray:
    """Mirror Stage-1 tokenization/encoding, never fit a new vocabulary."""
    config = json.loads((MODELS / "sequence_tokenizer_config.json").read_text(encoding="utf-8"))
    vocab = json.loads((MODELS / "sequence_vocabulary.json").read_text(encoding="utf-8"))
    if (config.get("standardization") != "NFKC + casefold"
            or config.get("sequence_length") != 60
            or config.get("vocabulary_size") != 3443
            or len(vocab) != 3443
            or config.get("unknown_token_index") != 1
            or vocab[:2] != ["[PAD]", "[UNK]"]
            or config.get("token_pattern") != r"(?u)\b\w+\b|[^\w\s]"):
        raise RuntimeError("Tokenizer berbeda dari konfigurasi UAS Tahap 1.")
    import unicodedata
    word_to_id = {token: idx for idx, token in enumerate(vocab)}
    output = np.zeros((len(texts), 60), dtype=np.int32)
    pattern = re.compile(config["token_pattern"])
    for row, value in enumerate(texts.astype(str)):
        tokens = pattern.findall(unicodedata.normalize("NFKC", value).casefold())
        if not tokens:
            raise RuntimeError("Satu SMS TEST kosong setelah tokenisasi.")
        ids = [word_to_id.get(token, 1) for token in tokens[:60]]
        output[row, :len(ids)] = ids
    return output


def evaluate_models(test: pd.DataFrame, preparation) -> tuple[dict, dict, list[list[int]], list[list[int]]]:
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    truth = (test["label"] == "spam").to_numpy(dtype=np.int32)

    tfidf = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    x_mlp = tfidf.transform(test["text"]).astype(np.float32).toarray()
    if x_mlp.shape != (len(test), 2000):
        raise RuntimeError("Fitur TF-IDF test tidak berukuran 2000.")
    mlp = tf.keras.models.load_model(MODELS / "mlp_baseline.keras", compile=False)
    if int(mlp.count_params()) != 130177:
        raise RuntimeError("Arsitektur MLP berubah.")
    mlp_prob = mlp.predict(x_mlp, batch_size=64, verbose=0).reshape(-1)
    del mlp, x_mlp
    tf.keras.backend.clear_session()

    sequence = encode_test(test["text"])
    gru = tf.keras.models.load_model(MODELS / "gru_attention_tuned.keras", compile=False)
    if int(gru.count_params()) != 246433:
        raise RuntimeError("Arsitektur GRU T3 berubah.")
    gru_prob = gru.predict(sequence, batch_size=64, verbose=0).reshape(-1)
    del gru, sequence
    tf.keras.backend.clear_session()

    if (mlp_prob.shape != truth.shape or gru_prob.shape != truth.shape
            or not np.isfinite(mlp_prob).all() or not np.isfinite(gru_prob).all()
            or (mlp_prob < 0).any() or (mlp_prob > 1).any()
            or (gru_prob < 0).any() or (gru_prob > 1).any()):
        raise RuntimeError("Probabilitas TEST tidak valid.")
    stage2 = preparation.load_script(ROOT / "scripts" / "07_train_gru.py", "_stage5_test_metrics")
    mpred = (mlp_prob >= THRESHOLD).astype(np.int32)
    gpred = (gru_prob >= THRESHOLD).astype(np.int32)
    mmetrics = stage2.classification_metrics(truth, mpred, mlp_prob)
    gmetrics = stage2.classification_metrics(truth, gpred, gru_prob)
    mcm = confusion_matrix(truth, mpred, labels=[0, 1]).tolist()
    gcm = confusion_matrix(truth, gpred, labels=[0, 1]).tolist()
    if (sum(map(sum, mcm)) != len(test) or sum(map(sum, gcm)) != len(test)
            or [sum(row) for row in mcm] != [int(np.sum(truth == 0)), int(np.sum(truth == 1))]
            or [sum(row) for row in gcm] != [int(np.sum(truth == 0)), int(np.sum(truth == 1))]):
        raise RuntimeError("Confusion matrix TEST tidak sesuai distribusi kelas.")
    return mmetrics, gmetrics, mcm, gcm


def write_outputs(report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    FIGURES.mkdir(parents=True, exist_ok=True)
    keys = ["accuracy", "precision_spam", "recall_spam", "f1_spam", "pr_auc_average_precision"]
    labels = ["Accuracy", "Precision\nSpam", "Recall\nSpam", "F1\nSpam", "PR-AUC"]
    mlp = report["test_metrics"]["mlp_uts"]
    gru = report["test_metrics"]["gru_t3_uas"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(keys))
    ax.bar(xs - .19, [mlp[k] for k in keys], width=.38, label="MLP UTS", color="#9aa9b9")
    ax.bar(xs + .19, [gru[k] for k in keys], width=.38, label="GRU T3 UAS", color="#216e8e")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.set_title("FINAL TEST — UTS MLP vs UAS GRU + Attention")
    ax.grid(axis="y", alpha=.16)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name, title in zip(axes, ("mlp_uts", "gru_t3_uas"), ("UTS MLP", "UAS T3 GRU+Attention")):
        mat = np.asarray(report["confusion_matrices"][name], dtype=int)
        ax.imshow(mat, cmap="Blues", norm=Normalize(vmin=0, vmax=max(1, int(mat.max()))))
        ax.set_xticks((0, 1), ("Ham", "Spam"))
        ax.set_yticks((0, 1), ("Ham", "Spam"))
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title(title)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.suptitle("Confusion matrix — held-out TEST (516 SMS)")
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=160)
    plt.close(fig)

    records = [
        {"model": name, **report["test_metrics"][name]}
        for name in ("mlp_uts", "gru_t3_uas")
    ]
    pd.DataFrame(records).to_csv(COMPARISON_CSV, index=False, encoding="utf-8-sig")
    # JSON is the authoritative result and is written LAST; no overwriting.
    with FINAL_RESULT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run_status() -> None:
    print("=" * 70)
    print("UAS TAHAP 5B - SAFE STATUS (TEST NOT READ)")
    print("=" * 70)
    print("Freeze tersedia:", FREEZE.is_file())
    print("Marker one-time tersedia:", RUN_MARKER.exists())
    print("Hasil final tersedia:", FINAL_RESULT.is_file())
    if FREEZE.is_file():
        prepared = load_preparation()
        lock, _ = verify_freeze(prepared)
        print("Protokol:", lock["status"])
        print("Model: MLP UTS vs T3 UAS | Threshold: 0.5")
    if FINAL_RESULT.is_file():
        result = json.loads(FINAL_RESULT.read_text(encoding="utf-8"))
        print("Status hasil:", result.get("status"))
    print("Perintah --status tidak membaca sms_test.csv.")


def run_evaluate(confirmation: str) -> None:
    if confirmation != "EVALUATE_LOCKED_TEST_ONCE":
        raise RuntimeError("Evaluasi TEST memerlukan --confirm EVALUATE_LOCKED_TEST_ONCE.")
    if not FREEZE.is_file():
        raise RuntimeError("Protokol 5A belum FROZEN; TEST tetap terkunci.")
    if RUN_MARKER.exists() or FINAL_RESULT.exists() or COMPARISON_CSV.exists() or COMPARISON_PNG.exists() or CONFUSION_PNG.exists():
        raise RuntimeError("Evaluasi TEST pernah dimulai/berkas hasil sudah ada. STOP, jangan evaluasi ulang.")
    prepared = load_preparation()
    lock, context = verify_freeze(prepared)  # Does NOT open TEST.
    if not (SPLITS / "sms_test.csv").is_file():
        raise FileNotFoundError("Berkas sms_test.csv tidak tersedia.")

    # Deliberate durable marker, created before the *first* TEST file read.
    # Do not delete this marker if evaluation fails or process is interrupted.
    with RUN_MARKER.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "status": "TEST_ACCESS_BEGAN_ONCE",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": prepared.sha256_file(FREEZE),
            "warning": "Do not remove marker or rerun; seek manual investigation if interrupted.",
        }, indent=2))
        stream.write("\n")
    print("ONE-TIME TEST ACCESS STARTED — marker written; NEVER repeat without audit.", flush=True)
    try:
        test, digest = read_and_validate_test(prepared, context, lock)
        mlp, gru, mlp_cm, gru_cm = evaluate_models(test, prepared)
        if prepared.hashes_for_freeze() != lock["input_sha256"]:
            raise RuntimeError("Artefak berubah selama evaluasi TEST.")
        report = {
            "status": "UAS_FINAL_TEST_EVALUATED_ONCE",
            "scope": "HELD_OUT_TEST_ONLY__AFTER_VALIDATION_PROTOCOL_FREEZE",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": prepared.sha256_file(FREEZE),
            "test_sha256": digest,
            "test_count": len(test),
            "test_class_counts": test["label"].value_counts().to_dict(),
            "seed": 42,
            "threshold": THRESHOLD,
            "models": {
                "mlp_uts": "models/mlp_baseline.keras",
                "gru_t3_uas": "models/gru_attention_tuned.keras",
            },
            "validation_metrics_frozen": {
                "mlp_uts": lock["validation_mlp_metrics"],
                "gru_t3_uas": lock["validation_t3_metrics"],
            },
            "test_metrics": {"mlp_uts": mlp, "gru_t3_uas": gru},
            "confusion_matrices": {"mlp_uts": mlp_cm, "gru_t3_uas": gru_cm},
            "difference_gru_minus_mlp": {
                key: round(float(gru[key] - mlp[key]), 6) for key in mlp
            },
            "test_selection_usage": "NONE; frozen models and threshold remain unchanged",
            "outputs": [
                str(COMPARISON_CSV.relative_to(ROOT)).replace("\\", "/"),
                str(COMPARISON_PNG.relative_to(ROOT)).replace("\\", "/"),
                str(CONFUSION_PNG.relative_to(ROOT)).replace("\\", "/"),
            ],
            "limitations": [
                "Results depend on this one held-out test split; no retraining or post-test tuning.",
                "Historical English SMS; unknown generalization to modern Indonesian SMS.",
                "Attention weights are not causal explanations.",
            ],
        }
        write_outputs(report)
    except Exception:
        print("Evaluasi dihentikan setelah akses TEST. Marker tetap ada; JANGAN rerun otomatis.", flush=True)
        raise

    print("=" * 70)
    print("UAS TAHAP 5B - FINAL TEST RESULT (SINGLE RUN)")
    print("=" * 70)
    print("Test rows:", report["test_count"])
    print("MLP UTS:", report["test_metrics"]["mlp_uts"])
    print("GRU T3:", report["test_metrics"]["gru_t3_uas"])
    print("Confusion MLP:", mlp_cm)
    print("Confusion GRU:", gru_cm)
    print("Hasil:", FINAL_RESULT.relative_to(ROOT))
    print("TEST sudah dievaluasi satu kali. Jangan tuning ulang menggunakan hasil ini.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi final test satu kali untuk model yang dikunci.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status", action="store_true", help="Hanya membaca status non-test.")
    modes.add_argument("--evaluate-test", action="store_true", help="Buka TEST setelah persetujuan eksplisit.")
    parser.add_argument("--confirm", default="", help="Token otorisasi manual untuk evaluasi.")
    args = parser.parse_args()
    if args.status:
        if args.confirm:
            parser.error("--status tidak membutuhkan --confirm.")
        run_status()
    else:
        run_evaluate(args.confirm)


if __name__ == "__main__":
    main()
