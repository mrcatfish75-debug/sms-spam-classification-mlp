r"""UAS Tahap 4B: selected T3 attention and validation-only error cases.

Run from the project root after 09_ablation_gru.py:
    .\.venv\Scripts\python.exe scripts\10_interpret_validation.py

Reads only TRAIN/VALIDATION artifacts and the validation SMS split; TEST stays locked.
Never interprets attention as proof that a word caused a prediction.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"
SPLITS = ROOT / "data" / "splits"
ABL_SCRIPT = ROOT / "scripts" / "09_ablation_gru.py"
ABL_SUMMARY = REPORTS / "uas_ablation_summary.json"
OUTPUT_JSON = REPORTS / "uas_validation_attention_analysis.json"
OUTPUT_CSV = REPORTS / "uas_validation_manual_review.csv"


def load_ablation():
    if not ABL_SCRIPT.is_file():
        raise FileNotFoundError("scripts/09_ablation_gru.py tidak ditemukan.")
    spec = importlib.util.spec_from_file_location("_uas_stage4_ablation", ABL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Tidak bisa memuat pemeriksa Tahap 4A.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def redact_text(text: str) -> str:
    """Best-effort masking; reports stay private even after masking."""
    text = re.sub(r"(?i)(?:https?://|www\.)\S+", "[URL]", str(text))
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"\d+", "[NUM]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:220] + ("…" if len(text) > 220 else "")


def safe_token(text: str) -> str:
    if text == "[UNK]":
        return text
    if re.search(r"\d", text):
        return "[NUM]"
    if "@" in text:
        return "[AT]"
    if text.lower() in {"http", "https", "www"}:
        return "[URL_PART]"
    return text[:28]


def main() -> None:
    print("=" * 70)
    print("UAS TAHAP 4B — ANALISIS VALIDATION + TOKEN ATTENTION")
    print("=" * 70)
    ablation = load_ablation()
    stage3, stage2, selection, signature, _, _, x_val, y_val, _ = ablation.validated_inputs()
    if not ABL_SUMMARY.is_file():
        raise FileNotFoundError("Selesaikan Tahap 4A: jalankan 09_ablation_gru.py dahulu.")
    ablation_report = ablation.read_json(ABL_SUMMARY)
    if (ablation_report.get("status") != "UAS_ABLATION_VALIDATION_ONLY_PASS"
            or ablation_report.get("test_status") != "LOCKED_NOT_EVALUATED"
            or ablation_report.get("stage4_input_signature") != signature):
        raise RuntimeError("Hasil ablation Tahap 4A belum PASS atau berbeda dengan model T3.")

    validation = pd.read_csv(SPLITS / "sms_validation.csv", keep_default_na=False, encoding="utf-8-sig")
    if not {"source_row", "label", "text"}.issubset(validation.columns):
        raise RuntimeError("Kolom validation asli tidak lengkap.")
    with np.load(ROOT / "data/processed/sequences_validation.npz", allow_pickle=False) as artifact:
        ids = artifact["source_row"].copy()
    if (len(validation) != len(x_val) or
            not np.array_equal(validation["source_row"].to_numpy(dtype=np.int64), ids)
            or not np.array_equal((validation["label"].str.lower() == "spam").to_numpy(dtype=np.int32), y_val)):
        raise RuntimeError("Baris/label validation berbeda dari sekuensial Tahap 1.")

    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    classifier = tf.keras.models.load_model(MODELS / "gru_attention_tuned.keras", compile=False)
    attention_reader = tf.keras.models.load_model(MODELS / "gru_attention_tuned_reader.keras", compile=False)
    probability = classifier.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    prediction = (probability >= stage3.THRESHOLD).astype(np.int32)
    metrics = stage2.classification_metrics(y_val, prediction, probability)
    for key, reported in selection["winner_validation_metrics"].items():
        if abs(metrics[key] - reported) > 2e-5:
            raise RuntimeError(f"Metrik {key} T3 tidak sesuai Tahap 3.")
    cm = confusion_matrix(y_val, prediction, labels=[0, 1])
    if int(cm.sum()) != len(y_val):
        raise RuntimeError("Confusion matrix tidak memuat seluruh validation.")

    scores = attention_reader.predict(x_val, batch_size=64, verbose=0)
    if (scores.shape != (len(x_val), 1, x_val.shape[1])
            or not np.allclose(scores.sum(axis=-1), 1, atol=1e-4)
            or np.abs(scores[:, 0, :][x_val == 0]).max(initial=0) > 1e-4):
        raise RuntimeError("Attention reader T3 gagal pemeriksaan masking.")

    margin = np.abs(probability - stage3.THRESHOLD)
    errors = np.flatnonzero(prediction != y_val)
    correct = np.flatnonzero(prediction == y_val)
    # ALL errors first, then enough correctly classified but relatively least
    # confident predictions to obtain five validation cases in total.
    ordered_errors = sorted(errors.tolist(), key=lambda i: (float(margin[i]), int(ids[i])))
    ordered_correct = sorted(correct.tolist(), key=lambda i: (float(margin[i]), int(ids[i])))
    count_cases = min(5, len(y_val))
    chosen = ordered_errors[:count_cases]
    chosen += ordered_correct[: count_cases - len(chosen)]
    if len(chosen) != count_cases or len(set(chosen)) != len(chosen):
        raise RuntimeError("Pemilihan lima contoh validation tidak valid.")

    config = ablation.read_json(MODELS / "sequence_tokenizer_config.json")
    vocab = json.loads((MODELS / "sequence_vocabulary.json").read_text(encoding="utf-8"))
    word_to_id = {word: i for i, word in enumerate(vocab)}
    pattern = config["token_pattern"]
    if config["sequence_length"] != x_val.shape[1] or len(vocab) != config["vocabulary_size"]:
        raise RuntimeError("Konfigurasi tokenizer berbeda dari Tahap 1.")

    cases = []
    csv_rows = []
    for k, index in enumerate(chosen, start=1):
        original_text = str(validation.iloc[index]["text"])
        tokens = re.findall(pattern, unicodedata.normalize("NFKC", original_text).casefold())
        encoded = np.zeros(x_val.shape[1], dtype=np.int32)
        encoded[: min(len(tokens), len(encoded))] = [
            word_to_id.get(token, 1) for token in tokens[: len(encoded)]
        ]
        if not np.array_equal(encoded, x_val[index]):
            raise RuntimeError(f"Tokenization mismatch pada source_row {ids[index]}; data tidak konsisten.")
        visible = min(len(tokens), x_val.shape[1])
        weights = scores[index, 0, :visible]
        top_positions = sorted(range(visible), key=lambda i: (-float(weights[i]), i))[:5]
        group = "incorrect" if prediction[index] != y_val[index] else "correct_lowest_margin"
        top_tokens = [
            {
                "position_1_based": int(pos + 1),
                "token_redacted": safe_token("[UNK]" if x_val[index, pos] == 1 else tokens[pos]),
                "weight": round(float(weights[pos]), 6),
            }
            for pos in top_positions
        ]
        item = {
            "case_id": f"V{k}",
            "source_row": int(ids[index]),
            "group": group,
            "actual": "spam" if y_val[index] else "ham",
            "predicted": "spam" if prediction[index] else "ham",
            "probability_spam": round(float(probability[index]), 6),
            "margin_from_threshold": round(float(margin[index]), 6),
            "token_count_before_truncation": len(tokens),
            "tokens_used": visible,
            "text_redacted_for_private_review": redact_text(original_text),
            "top_attention_tokens": top_tokens,
            "attention_weights_sum_to_one": bool(np.isclose(scores[index].sum(), 1, atol=1e-4)),
        }
        cases.append(item)
        csv_rows.append({
            "case_id": item["case_id"],
            "source_row": item["source_row"],
            "jenis_kasus": group,
            "label_aktual": item["actual"],
            "prediksi": item["predicted"],
            "probabilitas_spam": item["probability_spam"],
            "margin_dari_0_5": item["margin_from_threshold"],
            "teks_disamarkan": item["text_redacted_for_private_review"],
            "top_attention_ringkas": ", ".join(
                f"{tok['token_redacted']}:{tok['weight']:.3f}" for tok in top_tokens
            ),
            "review_label_valid_ya_tidak": "",
            "dugaan_penyebab_manual": "",
            "catatan_reviewer": "",
        })

    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Public-safe heatmap uses only token POSITIONS, not the raw SMS content.
    ncols = max(int(np.count_nonzero(x_val[idx])) for idx in chosen)
    heatmap = np.zeros((len(chosen), ncols), dtype=np.float32)
    for i, idx in enumerate(chosen):
        heatmap[i, :] = scores[idx, 0, :ncols]
    fig, ax = plt.subplots(figsize=(max(8, ncols / 4), 3.7))
    image = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(cases)), [
        f"{entry['case_id']} - {'salah' if entry['group'] == 'incorrect' else 'benar'}"
        for entry in cases
    ])
    ax.set_xlabel("Posisi token dalam sekuensial (teks tidak ditampilkan)")
    ax.set_ylabel("Kasus validation")
    ax.set_title("Distribusi bobot attention - model T3 (VALIDATION)")
    if ncols <= 25:
        ax.set_xticks(range(ncols), [str(i + 1) for i in range(ncols)])
    else:
        step = max(1, ncols // 12)
        ax.set_xticks(range(0, ncols, step), [str(i + 1) for i in range(0, ncols, step)])
    fig.colorbar(image, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(FIGURES / "attention_validation_cases.png", dpi=160)
    plt.close(fig)

    summary = {
        "status": "UAS_INTERPRET_VALIDATION_ONLY_PASS",
        "scope": "VALIDATION_ONLY__TEST_NOT_READ",
        "model": "Selected T3 from Stage 3; not modified or retrained",
        "stage4_input_signature": signature,
        "validation_count": len(y_val),
        "model_validation_metrics_reverified": metrics,
        "confusion_matrix_actual_0ham_1spam_predicted_0ham_1spam": cm.tolist(),
        "errors": int(len(errors)),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "cases_reported": len(cases),
        "genuine_misclassifications_in_report": int(sum(c["group"] == "incorrect" for c in cases)),
        "correct_lowest_margin_in_report": int(sum(c["group"] == "correct_lowest_margin" for c in cases)),
        "five_cases": cases,
        "attention_mask_valid": True,
        "manual_error_explanations": "PENDING_USER_REVIEW",
        "test_status": "LOCKED_NOT_EVALUATED",
        "privacy_note": "Redaction is best effort. Review CSV/JSON privately; do not git add outputs/reports/*.csv or *.json.",
        "interpretation_warning": "Attention weights show learned focus/weight distribution, not a causal explanation of the classification.",
    }
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validation: {len(y_val)} | actual misclassifications: {len(errors)}")
    print(f"False positive: {cm[0, 1]} | false negative: {cm[1, 0]}")
    print(f"Five reviewed candidates: {len(ordered_errors[:count_cases])} error(s) + {len(cases) - len(ordered_errors[:count_cases])} correctly classified low-margin SMS")
    print("Manual review CSV:", OUTPUT_CSV.relative_to(ROOT))
    print("Report JSON:", OUTPUT_JSON.relative_to(ROOT))
    print("Attention positions figure:", (FIGURES / "attention_validation_cases.png").relative_to(ROOT))
    print("TEST: LOCKED_NOT_EVALUATED")
    print("UAS TAHAP 4B PASS - lima contoh siap peninjauan manual.")


if __name__ == "__main__":
    main()
