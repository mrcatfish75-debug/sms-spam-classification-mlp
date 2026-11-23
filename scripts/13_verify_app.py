r"""Verify the frozen Gradio inference artifacts without touching dataset splits.

Windows PowerShell, from project root:
    .\.venv\Scripts\python.exe scripts\13_verify_app.py --check
    .\.venv\Scripts\python.exe scripts\13_verify_app.py --smoke

--check: checks SHA256, vocabulary and shape without TensorFlow.
--smoke: additionally runs two *synthetic* messages through loaded models.
Neither mode reads any training, validation, or test SMS CSV/NPZ.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.uas_inference import (  # noqa: E402
    FROZEN_INPUTS,
    THRESHOLD,
    encode_sms,
    frozen_artifacts,
    label_and_confidence,
    predict_gru,
    predict_mlp,
    sha256_file,
)


def perform_static_check() -> dict:
    artifact = frozen_artifacts()
    if (len(artifact.vocabulary) != 3443 or artifact.sequence_length != 60
            or artifact.vocabulary[:2] != ("[PAD]", "[UNK]")):
        raise RuntimeError("Konfigurasi vocabulary/sequence bukan hasil Tahap 1.")

    selection = json.loads(
        (ROOT / "outputs/reports/uas_tuning_selection.json").read_text(encoding="utf-8")
    )
    if (
        selection.get("status") != "UAS_TUNING_VALIDATION_ONLY_PASS"
        or selection.get("winner_trial") != "T3"
        or selection.get("winner_config") != {"gru_units": 64, "dropout": 0.2}
        or selection.get("selected_classifier") != "models/gru_attention_tuned.keras"
        or selection.get("selected_classifier_sha256")
            != sha256_file(ROOT / "models/gru_attention_tuned.keras")
        or selection.get("selected_attention_reader_sha256")
            != sha256_file(ROOT / "models/gru_attention_tuned_reader.keras")
    ):
        raise RuntimeError("Model T3 tidak cocok dengan laporan tuning Tahap 3.")

    sample = encode_sms("HI 😊")
    if (sample.tokens != ("hi", "😊") or sample.sequence.shape != (1, 60)
            or sample.sequence.dtype.name != "int32"
            or (sample.sequence[0, 2:] != 0).any()):
        raise RuntimeError("Pemeriksaan tokenisasi Unicode atau post-padding gagal.")
    if label_and_confidence(0.5)[0] != "Spam" or label_and_confidence(0.499)[0] != "Ham":
        raise RuntimeError("Threshold 0.5 berubah.")

    return {
        "status": "UAS_STAGE6_STATIC_VERIFIED",
        "scope": "SYNTHETIC_AND_METADATA_ONLY__NO_DATASET_READ",
        "frozen_trial": "T3",
        "threshold": THRESHOLD,
        "vocabulary_size": len(artifact.vocabulary),
        "sequence_length": artifact.sequence_length,
        "verified_model_inputs": list(FROZEN_INPUTS),
    }


def perform_smoke(report: dict) -> dict:
    # These two examples are authored for smoke testing, not sampled from UCI.
    examples = (
        "Hi, can we meet for coffee tomorrow afternoon?",
        "Congratulations! Claim your free prize by calling our number today!",
    )
    scores = []
    for message in examples:
        mlp = predict_mlp(message)
        gru = predict_gru(message, with_attention=True)
        if not (math.isfinite(mlp) and 0 <= mlp <= 1
                and math.isfinite(gru["probability_spam"])
                and 0 <= gru["probability_spam"] <= 1):
            raise RuntimeError("Model menghasilkan probabilitas tidak valid.")
        if not gru["top_attention"] or len(gru["top_attention"]) > 5:
            raise RuntimeError("Output attention tidak terisi dengan benar.")
        scores.append({
            "case": f"SYNTHETIC_{len(scores) + 1}",
            "mlp_probability_spam": round(mlp, 6),
            "gru_t3_probability_spam": round(gru["probability_spam"], 6),
            "attention_count": len(gru["top_attention"]),
        })
    report["status"] = "UAS_STAGE6_SMOKE_PASS"
    report["synthetic_case_count"] = len(examples)
    report["synthetic_results"] = scores
    report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()

    report_path = ROOT / "outputs/reports/uas_stage6_app_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Smoke report: {report_path.relative_to(ROOT)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Stage 6 verification")
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument("--check", action="store_true", help="Only validate frozen artifacts")
    flags.add_argument("--smoke", action="store_true", help="Predict synthetic messages (never test)")
    args = parser.parse_args()

    print("=" * 66)
    print("UAS TAHAP 6 - STATIC CHECK" if args.check else "UAS TAHAP 6 - SYNTHETIC SMOKE")
    print("=" * 66)
    report = perform_static_check()
    print("Freeze T3/MLP SHA256: PASS")
    print("Tokenizer 3443 vocabulary / 60 tokens: PASS")
    print("Threshold 0.5: PASS")
    if args.smoke:
        report = perform_smoke(report)
        for entry in report["synthetic_results"]:
            print(entry)
    print("STATUS:", report["status"])
    print("NO TRAIN/VALIDATION/TEST DATASET WAS READ")


if __name__ == "__main__":
    main()
