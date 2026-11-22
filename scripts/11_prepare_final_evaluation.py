r"""UAS Tahap 5A: review lima kesalahan MLP dan kunci protokol, VALIDATION ONLY.

Penggunaan (PowerShell, dari root proyek):
    .\.venv\Scripts\python.exe scripts\11_prepare_final_evaluation.py --prepare
    # Review manual 5 kasus pada CSV yang dihasilkan.
    .\.venv\Scripts\python.exe scripts\11_prepare_final_evaluation.py --freeze --confirm FREEZE_UTS_UAS

KODE INI TIDAK MEMBUKA / MEMBACA sms_test.csv, TIDAK MELATIH ULANG,
DAN TIDAK MEMODIFIKASI MODEL, TOKENIZER, ATAU OUTPUT TAHAP 1-4.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from operator import index
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "outputs" / "reports"
STAGE4_SCRIPT = ROOT / "scripts" / "09_ablation_gru.py"
INTERPRET_SCRIPT = ROOT / "scripts" / "10_interpret_validation.py"
FIVE_ERRORS_CSV = REPORTS / "uas_mlp_five_errors_manual_review.csv"
AUDIT_REPORT = REPORTS / "uas_stage5_pretest_audit.json"
FREEZE_REPORT = REPORTS / "uas_final_protocol_lock.json"
FINAL_TEST_RESULT = REPORTS / "uas_final_test_results.json"
FINAL_TEST_ONCE_MARKER = REPORTS / "uas_final_test_once.lock"
THRESHOLD = 0.5

REQUIRED_FREEZE_FILES = (
    "data/processed/sms_en_clean.csv",
    "data/processed/sequences_train.npz",
    "data/processed/sequences_validation.npz",
    "data/splits/sms_train.csv",
    "data/splits/sms_validation.csv",
    "models/mlp_baseline.keras",
    "models/tfidf_vectorizer.joblib",
    "models/gru_attention_tuned.keras",
    "models/gru_attention_tuned_reader.keras",
    "models/word2vec_embedding_matrix.npy",
    "models/sequence_vocabulary.json",
    "models/sequence_tokenizer_config.json",
    "outputs/reports/split_tfidf_manifest.json",
    "outputs/reports/hasil_validasi_mlp.json",
    "outputs/reports/uas_sequence_word2vec_manifest.json",
    "outputs/reports/hasil_validasi_gru_attention.json",
    "outputs/reports/uas_tuning_selection.json",
    "outputs/reports/uas_ablation_summary.json",
    "outputs/reports/uas_validation_attention_analysis.json",
    "outputs/reports/uas_validation_manual_review.csv",
    "outputs/reports/uas_mlp_five_errors_manual_review.csv",
    "scripts/03_split_tfidf.py",
    "scripts/04_train_mlp.py",
    "scripts/05_analyze_validation_errors.py",
    "scripts/06_prepare_sequences.py",
    "scripts/07_train_gru.py",
    "scripts/08_tune_gru.py",
    "scripts/09_ablation_gru.py",
    "scripts/10_interpret_validation.py",
    "scripts/11_prepare_final_evaluation.py",
    "scripts/12_evaluate_final_test.py",
)


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Berkas prasyarat hilang: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Laporan prasyarat hilang: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_script(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Script prasyarat hilang: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak dapat memuat script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_no_final_test_started() -> None:
    if FINAL_TEST_ONCE_MARKER.exists() or FINAL_TEST_RESULT.exists():
        raise RuntimeError(
            "Evaluasi final TEST sudah dimulai/selesai. Tidak boleh menyiapkan ulang "
            "atau mengubah protokol menggunakan hasil TEST."
        )


def verify_stage4_review(report: dict, selected_signature: dict) -> pd.DataFrame:
    if (report.get("status") != "UAS_INTERPRET_VALIDATION_ONLY_PASS"
            or report.get("test_status") != "LOCKED_NOT_EVALUATED"
            or report.get("stage4_input_signature") != selected_signature
            or report.get("errors") != 2 or report.get("false_positives") != 0
            or report.get("false_negatives") != 2
            or report.get("genuine_misclassifications_in_report") != 2
            or report.get("correct_lowest_margin_in_report") != 3
            or report.get("attention_mask_valid") is not True):
        raise RuntimeError("Laporan analisis attention Tahap 4 tidak cocok dengan T3.")
    expected = report.get("five_cases", [])
    if len(expected) != 5:
        raise RuntimeError("Jumlah kandidat manual review T3 harus tepat lima.")
    path = REPORTS / "uas_validation_manual_review.csv"
    if not path.is_file():
        raise FileNotFoundError("Review manual Tahap 4 belum tersedia.")
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    required = {
        "case_id", "source_row", "jenis_kasus", "label_aktual", "prediksi",
        "probabilitas_spam", "margin_dari_0_5", "teks_disamarkan",
        "review_label_valid_ya_tidak", "dugaan_penyebab_manual", "catatan_reviewer",
    }
    if len(frame) != 5 or not required.issubset(frame.columns):
        raise RuntimeError("CSV review T3 harus terdiri atas lima baris dan seluruh kolom wajib.")
    for index, (row, case) in enumerate(zip(frame.to_dict("records"), expected)):
        if (row["case_id"] != case["case_id"]
                or int(row["source_row"]) != case["source_row"]
                or row["jenis_kasus"] != case["group"]
                or row["label_aktual"] != case["actual"]
                or row["prediksi"] != case["predicted"]
                or row["teks_disamarkan"] != case["text_redacted_for_private_review"]
                or abs(float(row["probabilitas_spam"]) - case["probability_spam"]) > 1e-6
                or abs(float(row["margin_dari_0_5"]) - case["margin_from_threshold"]) > 1e-6):
            raise RuntimeError(f"CSV review T3 {case['case_id']} berbeda dari laporan otomatis.")
# Keputusan final review: V1 belum pasti; V2 dan V3 = YA.
# V4 dan V5 tetap belum dapat dipastikan.
        # Keputusan final review:
        # V1 belum pasti; V2 dan V3 = YA; V4 dan V5 belum pasti.
        expected_status = "YA" if index in (1, 2) else ""

        if str(row["review_label_valid_ya_tidak"]).strip().upper() != expected_status:
            raise RuntimeError(
                f"Status review {case['case_id']} tidak sesuai keputusan kelompok."
            )

        if not str(row["dugaan_penyebab_manual"]).strip() or not str(row["catatan_reviewer"]).strip():
            raise RuntimeError(
                f"Catatan review {case['case_id']} belum lengkap."
            )

    return frame      



def compute_validation_context() -> dict:
    """Read ONLY train/validation inputs. Test file never opened here."""
    ablation = load_script(STAGE4_SCRIPT, "_stage5_readonly_ablation")
    stage3, stage2, selection, signature, _, _, x_val, y_val, _ = ablation.validated_inputs()
    if selection.get("winner_trial") != "T3" or selection.get("winner_config") != {
        "gru_units": 64, "dropout": 0.2,
    }:
        raise RuntimeError("Model terpilih bukan T3 GRU64/dropout0.20.")
    ablation_report = read_json(REPORTS / "uas_ablation_summary.json")
    if (ablation_report.get("status") != "UAS_ABLATION_VALIDATION_ONLY_PASS"
            or ablation_report.get("test_status") != "LOCKED_NOT_EVALUATED"
            or ablation_report.get("stage4_input_signature") != signature):
        raise RuntimeError("Ablation Tahap 4A tidak PASS atau berbeda dari T3.")
    attention_report = read_json(REPORTS / "uas_validation_attention_analysis.json")
    verify_stage4_review(attention_report, signature)

    split_manifest = read_json(REPORTS / "split_tfidf_manifest.json")
    base_result = read_json(REPORTS / "hasil_validasi_mlp.json")
    if (split_manifest.get("status") != "SPLIT_TFIDF_PASS"
            or split_manifest.get("total_final") != 5159
            or split_manifest.get("train", {}).get("n") != 4127
            or split_manifest.get("validation", {}).get("n") != 516
            or split_manifest.get("test_locked", {}).get("n") != 516):
        raise RuntimeError("Split UTS tidak sesuai manifest final 5159 / 4127 / 516 / 516.")
    if sha256_file(ROOT / "data/processed/sms_en_clean.csv") != split_manifest.get("source_sha256"):
        raise RuntimeError("Dataset bersih berubah dari sumber yang tercatat dalam split manifest UTS.")
    if (base_result.get("status") != "MLP_TRAINED_VALIDATION_ONLY"
            or base_result.get("test_status") != "LOCKED_NOT_EVALUATED"
            or float(base_result.get("threshold", -1)) != THRESHOLD):
        raise RuntimeError("Laporan MLP UTS berubah atau tidak valid.")

    validation = pd.read_csv(SPLITS / "sms_validation.csv", encoding="utf-8-sig", keep_default_na=False)
    train = pd.read_csv(SPLITS / "sms_train.csv", encoding="utf-8-sig", keep_default_na=False)
    for name, frame, entry in (("train", train, split_manifest["train"]),
                               ("validation", validation, split_manifest["validation"])):
        if (len(frame) != entry["n"] or not {"source_row", "label", "text"}.issubset(frame.columns)
                or frame["source_row"].duplicated().any()):
            raise RuntimeError(f"Partisi {name} berubah.")
        if frame["label"].value_counts().to_dict() != entry["class_count"]:
            raise RuntimeError(f"Distribusi kelas {name} berbeda dari manifest.")

    with np.load(ROOT / "data/processed/sequences_validation.npz", allow_pickle=False) as sequence:
        seq_ids = sequence["source_row"]
        seq_labels = sequence["y"]
    if (not np.array_equal(seq_ids, validation["source_row"].to_numpy())
            or not np.array_equal(seq_labels, (validation["label"] == "spam").astype(np.int8).to_numpy())
            or not np.array_equal(seq_labels, y_val)):
        raise RuntimeError("Urutan/label validation MLP berbeda dari sekuens GRU.")

    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    x_mlp = vectorizer.transform(validation["text"]).astype(np.float32).toarray()
    if x_mlp.shape != (516, 2000):
        raise RuntimeError("Fitur TF-IDF validation harus 516 x 2000.")
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    mlp = tf.keras.models.load_model(MODELS / "mlp_baseline.keras", compile=False)
    if int(mlp.count_params()) != 130177:
        raise RuntimeError("Jumlah parameter MLP berbeda dari baseline 130177.")
    mlp_probability = mlp.predict(x_mlp, batch_size=64, verbose=0).reshape(-1)
    del mlp, x_mlp
    tf.keras.backend.clear_session()
    gru = tf.keras.models.load_model(MODELS / "gru_attention_tuned.keras", compile=False)
    if int(gru.count_params()) != 246433:
        raise RuntimeError("Jumlah parameter GRU T3 berbeda dari 246433.")
    gru_probability = gru.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    del gru
    tf.keras.backend.clear_session()

    truth = (validation["label"] == "spam").astype(np.int32).to_numpy()
    if not np.isfinite(mlp_probability).all() or not np.isfinite(gru_probability).all():
        raise RuntimeError("Probabilitas validation tidak finite.")
    mlp_pred = (mlp_probability >= THRESHOLD).astype(np.int32)
    gru_pred = (gru_probability >= THRESHOLD).astype(np.int32)
    cm_mlp = confusion_matrix(truth, mlp_pred, labels=[0, 1]).tolist()
    cm_gru = confusion_matrix(truth, gru_pred, labels=[0, 1]).tolist()
    if (cm_mlp != [[449, 3], [2, 62]]
            or cm_mlp != base_result.get("confusion_matrix_validation_labels_0ham_1spam")
            or cm_gru != [[452, 0], [2, 62]]
            or cm_gru != attention_report.get("confusion_matrix_actual_0ham_1spam_predicted_0ham_1spam")):
        raise RuntimeError(f"Confusion matrix berubah: MLP={cm_mlp}; T3={cm_gru}.")
    mlp_metrics = stage2.classification_metrics(truth, mlp_pred, mlp_probability)
    gru_metrics = stage2.classification_metrics(truth, gru_pred, gru_probability)
    for name, metrics, original in (
        ("MLP", mlp_metrics, base_result["validation_mlp"]),
        ("T3", gru_metrics, selection["winner_validation_metrics"]),
    ):
        for key, recorded in original.items():
            if abs(float(metrics[key]) - float(recorded)) > 2e-5:
                raise RuntimeError(f"Metrik validation {name}/{key} tidak konsisten.")
    incorrect_mlp = np.flatnonzero(mlp_pred != truth)
    incorrect_gru = np.flatnonzero(gru_pred != truth)
    if len(incorrect_mlp) != 5 or len(incorrect_gru) != 2:
        raise RuntimeError("Jumlah prediksi salah MLP/T3 berubah.")
    return {
        "validation": validation,
        "mlp_probability": mlp_probability,
        "mlp_pred": mlp_pred,
        "gru_pred": gru_pred,
        "incorrect_mlp": incorrect_mlp,
        "incorrect_gru": incorrect_gru,
        "mlp_metrics": mlp_metrics,
        "gru_metrics": gru_metrics,
        "cm_mlp": cm_mlp,
        "cm_gru": cm_gru,
        "selection": selection,
        "split_manifest": split_manifest,
        "signature_stage4": signature,
    }


def expected_five_error_rows(context: dict) -> list[dict]:
    interpretation = load_script(INTERPRET_SCRIPT, "_stage5_redaction_only")
    validation = context["validation"]
    truth = (validation["label"] == "spam").to_numpy(dtype=np.int8)
    rows = []
    ordered = sorted(context["incorrect_mlp"], key=lambda i: int(validation.iloc[int(i)]["source_row"]))
    for num, idx in enumerate(ordered, start=1):
        src = validation.iloc[int(idx)]
        predicted = int(context["mlp_pred"][int(idx)])
        rows.append({
            "case_id": f"M{num}",
            "model": "MLP_UTS_TFIDF",
            "source_row": int(src["source_row"]),
            "jenis_kasus": "false_positive" if not truth[int(idx)] else "false_negative",
            "label_aktual": str(src["label"]),
            "prediksi": "spam" if predicted else "ham",
            "probabilitas_spam": round(float(context["mlp_probability"][int(idx)]), 6),
            "teks_disamarkan": interpretation.redact_text(str(src["text"])),
            "review_label_valid_ya_tidak": "",
            "dugaan_penyebab_manual": "",
            "catatan_reviewer": "",
        })
    return rows


def validate_five_error_file(expected: list[dict], require_review: bool) -> pd.DataFrame:
    frame = pd.read_csv(FIVE_ERRORS_CSV, encoding="utf-8-sig", keep_default_na=False)
    if len(frame) != 5 or not set(expected[0]).issubset(frame.columns):
        raise RuntimeError("CSV lima kesalahan MLP tidak memiliki lima baris/kolom yang benar.")
    protected = tuple(k for k in expected[0] if k not in (
        "review_label_valid_ya_tidak", "dugaan_penyebab_manual", "catatan_reviewer"
    ))
    for row, original in zip(frame.to_dict("records"), expected):
        for key in protected:
            if key == "probabilitas_spam":
                same = abs(float(row[key]) - float(original[key])) <= 1e-6
            else:
                same = str(row[key]) == str(original[key])
            if not same:
                raise RuntimeError(f"Baris {original['case_id']} kolom {key} berubah.")
        status = str(row["review_label_valid_ya_tidak"]).strip().upper()
        if status not in ("", "YA", "TIDAK"):
            raise RuntimeError(f"Kolom validitas {original['case_id']} harus YA/TIDAK/kosong.")
        if require_review and (
            not str(row["dugaan_penyebab_manual"]).strip()
            or not str(row["catatan_reviewer"]).strip()
        ):
            raise RuntimeError(f"Catatan MLP {original['case_id']} belum diisi; jangan freeze.")
    return frame


def hashes_for_freeze() -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in REQUIRED_FREEZE_FILES}


def write_json_new(path: Path, content: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Tidak menimpa artefak yang sudah ada: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def run_prepare() -> None:
    ensure_no_final_test_started()
    if FREEZE_REPORT.exists():
        raise RuntimeError("Protokol sudah FROZEN. Tidak boleh menyiapkan ulang kasus.")
    ctx = compute_validation_context()
    expected = expected_five_error_rows(ctx)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if FIVE_ERRORS_CSV.is_file():
        validate_five_error_file(expected, require_review=False)
        print("CSV review lima kesalahan sudah ada; DIJAGA, tidak ditimpa.")
    else:
        pd.DataFrame(expected).to_csv(FIVE_ERRORS_CSV, index=False, encoding="utf-8-sig")
        print("CSV review lima kesalahan dibuat; tiga kolom perlu diisi manusia.")
    print("=" * 70)
    print("UAS TAHAP 5A - VALIDATION-ONLY PREPARE")
    print("=" * 70)
    print("MLP UTS: 5 kesalahan (3 FP, 2 FN) | GRU T3: 2 kesalahan (0 FP, 2 FN)")
    print("Metrik MLP validation:", ctx["mlp_metrics"])
    print("Metrik T3 validation:", ctx["gru_metrics"])
    print("MLP source_row:", [case["source_row"] for case in expected])
    print("T3 source_row:", [int(ctx["validation"].iloc[i]["source_row"]) for i in ctx["incorrect_gru"]])
    print("Review:", FIVE_ERRORS_CSV.relative_to(ROOT))
    print("TEST TIDAK DIBACA. JANGAN jalankan 12_evaluate_final_test.py dahulu.")


def run_freeze(confirmation: str) -> None:
    if confirmation != "FREEZE_UTS_UAS":
        raise RuntimeError("Freeze memerlukan --confirm FREEZE_UTS_UAS.")
    ensure_no_final_test_started()
    if FREEZE_REPORT.exists():
        raise RuntimeError("Protokol sudah terkunci. Tidak boleh membuat ulang freeze.")
    ctx = compute_validation_context()
    if not FIVE_ERRORS_CSV.is_file():
        raise RuntimeError("Jalankan --prepare dan review lima kesalahan MLP terlebih dahulu.")
    expected = expected_five_error_rows(ctx)
    review = validate_five_error_file(expected, require_review=True)
    signature = hashes_for_freeze()
    report = {
        "status": "UAS_FINAL_EVALUATION_PROTOCOL_FROZEN",
        "scope": "VALIDATION_ONLY__TEST_NOT_READ",
        "selected_trial": "T3",
        "model_uts": "models/mlp_baseline.keras",
        "model_uas": "models/gru_attention_tuned.keras",
        "selected_model": "Word2Vec train-only -> GRU64 -> AdditiveAttention -> Dense16 -> Sigmoid",
        "threshold": THRESHOLD,
        "test_expected_count_from_existing_split_manifest": 516,
        "split_seed": 42,
        "class_mapping": {"ham": 0, "spam": 1},
        "validation_mlp_metrics": ctx["mlp_metrics"],
        "validation_t3_metrics": ctx["gru_metrics"],
        "validation_confusion_matrix_mlp": ctx["cm_mlp"],
        "validation_confusion_matrix_t3": ctx["cm_gru"],
        "reviewed_mlp_misclassifications": len(review),
        "reviewed_t3_misclassifications": 2,
        "five_error_case_ids": [item["source_row"] for item in expected],
        "input_sha256": signature,
        "test_status": "LOCKED_NOT_EVALUATED",
        "one_time_test_command": "python scripts/12_evaluate_final_test.py --evaluate-test --confirm EVALUATE_LOCKED_TEST_ONCE",
        "policy": [
            "No training, fitting, tuning, rebalancing, threshold selection, or model switching using TEST.",
            "Evaluate existing UTS MLP and selected UAS T3 on the SAME held-out TEST.",
            "Metrics: accuracy, precision-spam, recall-spam, F1-spam, PR-AUC, confusion matrix.",
            "TEST can be opened only in explicitly confirmed evaluation mode.",
            "All artifacts, including this protocol, must stay immutable after opening TEST.",
        ],
    }
    write_json_new(FREEZE_REPORT, report)
    print("=" * 70)
    print("UAS TAHAP 5A - FINAL EVALUATION PROTOCOL FROZEN")
    print("=" * 70)
    print("Review MLP lima kasus: lengkap | Review T3 lima kasus: lengkap")
    print("Model dan tokenizer dicatat melalui SHA256; data TEST BELUM DIBACA.")
    print("Freeze:", FREEZE_REPORT.relative_to(ROOT))
    print("Status:", report["status"])
    print("STOP setelah ini; minta pemeriksaan hasil sebelum membuka TEST.")


def main() -> None:
    parser = argparse.ArgumentParser(description="UAS Tahap 5A: audit validation dan freeze tanpa test.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Buat 5 kasus MLP tanpa menimpa review lama.")
    mode.add_argument("--freeze", action="store_true", help="Kunci protokol setelah review MLP selesai.")
    parser.add_argument("--confirm", default="", help="Persetujuan eksplisit untuk freeze.")
    args = parser.parse_args()
    if args.prepare:
        if args.confirm:
            parser.error("--prepare tidak membutuhkan --confirm.")
        run_prepare()
    else:
        run_freeze(args.confirm)


if __name__ == "__main__":
    main()
