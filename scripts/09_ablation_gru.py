r"""UAS Tahap 4A: fair ablations of the selected T3 GRU64+attention model.

Run from the project root using Python 3.13:
    .\.venv\Scripts\python.exe scripts\09_ablation_gru.py

Uses only TRAIN and VALIDATION. TEST is deliberately not read.
Each new ablation is trained in a separate process for 8 GB RAM laptops.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
ABL_MODELS = MODELS / "uas_ablation"
ABL_REPORTS = REPORTS / "uas_ablation"
STAGE3_SCRIPT = ROOT / "scripts" / "08_tune_gru.py"
TUNING_SUMMARY = REPORTS / "uas_tuning_selection.json"
SELECTED_MODEL = MODELS / "gru_attention_tuned.keras"
SELECTED_ATTENTION = MODELS / "gru_attention_tuned_reader.keras"
ABLATION_SUMMARY = REPORTS / "uas_ablation_summary.json"

SEED = 42
TRIALS = {
    "A1_no_attention": {
        "attention": False, "initialization": "train_only_word2vec"
    },
    "A2_no_word2vec": {
        "attention": True, "initialization": "seeded_random_uniform"
    },
}
EXPECTED_WINNER = "T3"
EXPECTED_CONFIG = {"gru_units": 64, "dropout": 0.2}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Prasyarat tidak ditemukan: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Laporan tidak ditemukan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage3():
    if not STAGE3_SCRIPT.is_file():
        raise FileNotFoundError("scripts/08_tune_gru.py tidak ditemukan (Tahap 3).")
    spec = importlib.util.spec_from_file_location("_uas_stage3_compat", STAGE3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Tidak bisa memuat pemeriksa Tahap 3.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, expected in (
        ("SEED", 42), ("DENSE_UNITS", 16), ("LEARNING_RATE", 0.001),
        ("BATCH_SIZE", 32), ("MAX_EPOCHS", 15), ("PATIENCE", 3),
        ("THRESHOLD", 0.50),
    ):
        if getattr(module, name, object()) != expected:
            raise RuntimeError(f"Konfigurasi Tahap 3 {name} berubah.")
    if module.TRIALS.get(EXPECTED_WINNER) != EXPECTED_CONFIG:
        raise RuntimeError("Definisi T3 Tahap 3 tidak sesuai eksperimen terkunci.")
    return module


def validated_inputs():
    stage3 = load_stage3()
    stage2, _, signature, x_train, y_train, x_val, y_val, embedding = stage3.validated_inputs()
    selection = read_json(TUNING_SUMMARY)
    if (selection.get("status") != "UAS_TUNING_VALIDATION_ONLY_PASS"
            or selection.get("scope") != "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ"
            or selection.get("test_status") != "LOCKED_NOT_EVALUATED"):
        raise RuntimeError("Tahap 3 belum PASS atau data test tidak terkunci.")
    if selection.get("winner_trial") != EXPECTED_WINNER:
        raise RuntimeError("Model terpilih berubah, bukan T3; perlu desain ablation ulang.")
    if selection.get("winner_config") != EXPECTED_CONFIG:
        raise RuntimeError("Konfigurasi model T3 terpilih tidak sesuai GRU64/dropout0.20.")
    if selection.get("winner_parameters_total") != 246433:
        raise RuntimeError("Arsitektur T3 berbeda dari baseline ablation yang ditetapkan.")
    if selection.get("input_signature") != signature:
        raise RuntimeError("Input Tahap 1–3 berubah sejak model T3 dipilih.")
    if selection.get("selected_classifier") != "models/gru_attention_tuned.keras":
        raise RuntimeError("Lokasi model T3 terpilih tidak sesuai.")
    if selection.get("selected_attention_reader") != "models/gru_attention_tuned_reader.keras":
        raise RuntimeError("Lokasi attention reader T3 tidak sesuai.")
    if sha256_file(SELECTED_MODEL) != selection.get("selected_classifier_sha256"):
        raise RuntimeError("Bobot classifier T3 telah berubah.")
    if sha256_file(SELECTED_ATTENTION) != selection.get("selected_attention_reader_sha256"):
        raise RuntimeError("Bobot pembaca attention T3 telah berubah.")
    signature_4 = {
        "stage3_input_signature": signature,
        "stage3_selection_sha256": sha256_file(TUNING_SUMMARY),
        "selected_classifier_sha256": sha256_file(SELECTED_MODEL),
        "selected_attention_reader_sha256": sha256_file(SELECTED_ATTENTION),
    }
    return stage3, stage2, selection, signature_4, x_train, y_train, x_val, y_val, embedding


def configure_tf():
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.keras.utils.set_random_seed(SEED)
    tf.keras.backend.clear_session()
    return tf


def build_ablation(tf, embedding: np.ndarray, length: int, trial: str):
    definition = TRIALS[trial]
    if definition["initialization"] == "seeded_random_uniform":
        rng = np.random.default_rng(SEED)
        initialized_embedding = rng.uniform(-0.05, 0.05, embedding.shape).astype(np.float32)
        initialized_embedding[0] = 0.0
    else:
        initialized_embedding = embedding

    tokens = tf.keras.Input(shape=(length,), dtype="int32", name="token_ids")
    vectors = tf.keras.layers.Embedding(
        input_dim=embedding.shape[0], output_dim=embedding.shape[1],
        weights=[initialized_embedding], mask_zero=True, trainable=True,
        name="word2vec_embedding" if definition["attention"] is False else "input_embedding",
    )(tokens)
    sequences, state = tf.keras.layers.GRU(
        64, return_sequences=True, return_state=True, name="gru_encoder",
    )(vectors)
    attention_reader = None
    if definition["attention"]:
        query = tf.keras.layers.Reshape((1, 64), name="attention_query")(state)
        context, scores = tf.keras.layers.AdditiveAttention(
            use_scale=True, dropout=0.0, name="token_attention",
        )([query, sequences], return_attention_scores=True)
        features = tf.keras.layers.Flatten(name="attention_context")(context)
        attention_reader = tf.keras.Model(tokens, scores, name="ablation_attention_reader")
    else:
        # Replace attention-pooled context with the final GRU state of the same width.
        features = state
    features = tf.keras.layers.Dense(16, activation="relu", name="hidden_16")(features)
    features = tf.keras.layers.Dropout(0.20, name="dropout_020")(features)
    prob = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(features)
    classifier = tf.keras.Model(tokens, prob, name="sms_spam_gru_ablation")
    return classifier, attention_reader


def model_path(trial: str) -> Path:
    return ABL_MODELS / f"{trial}.keras"


def report_path(trial: str) -> Path:
    return ABL_REPORTS / f"{trial}_validation.json"


def valid_saved_trial(trial: str, signature: dict) -> dict:
    entry = read_json(report_path(trial))
    if (entry.get("status") != "ABLATION_TRIAL_PASS"
            or entry.get("trial") != trial
            or entry.get("config") != TRIALS[trial]
            or entry.get("stage4_input_signature") != signature
            or entry.get("test_status") != "LOCKED_NOT_EVALUATED"):
        raise RuntimeError(f"Laporan ablation {trial} tidak cocok dengan tahap sebelumnya.")
    if sha256_file(model_path(trial)) != entry.get("classifier_sha256"):
        raise RuntimeError(f"Bobot ablation {trial} rusak atau berubah.")
    if "validation_metrics" not in entry:
        raise RuntimeError(f"Metrik {trial} tidak lengkap.")
    return entry


def train_one(trial: str) -> None:
    stage3, stage2, _, signature, x_train, y_train, x_val, y_val, embedding = validated_inputs()
    print(f"\n{'=' * 68}\nUAS TAHAP 4A — {trial} (VALIDATION ONLY)\n{'=' * 68}", flush=True)
    tf = configure_tf()
    model, reader = build_ablation(tf, embedding, x_train.shape[1], trial)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=stage3.LEARNING_RATE),
        loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    model.summary()
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        batch_size=stage3.BATCH_SIZE, epochs=stage3.MAX_EPOCHS, shuffle=True, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=stage3.PATIENCE, restore_best_weights=True,
        )],
    )
    prob = model.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    if not np.isfinite(prob).all():
        raise RuntimeError(f"Model {trial} memiliki probabilitas tidak finite.")
    predicted = (prob >= stage3.THRESHOLD).astype(np.int32)
    metrics = stage2.classification_metrics(y_val, predicted, prob)
    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(y_val, predicted, labels=[0, 1])
    if int(matrix.sum()) != len(y_val):
        raise RuntimeError("Ukuran confusion matrix tidak sesuai validation.")
    probe = x_val[: min(32, len(x_val))]
    if reader is not None:
        weights = reader.predict(probe, verbose=0)
        if (weights.shape != (len(probe), 1, x_val.shape[1])
                or not np.allclose(weights.sum(axis=-1), 1, atol=1e-4)
                or np.abs(weights[:, 0, :][probe == 0]).max(initial=0) > 1e-4):
            raise RuntimeError(f"Masking attention {trial} gagal.")

    ABL_MODELS.mkdir(parents=True, exist_ok=True)
    ABL_REPORTS.mkdir(parents=True, exist_ok=True)
    saved = model_path(trial)
    model.save(saved)
    restored = tf.keras.models.load_model(saved, compile=False)
    if not np.allclose(restored.predict(probe, verbose=0),
                       model.predict(probe, verbose=0), atol=1e-6):
        raise RuntimeError(f"Reload model {trial} tidak sama dengan model asli.")

    epoch_data = pd.DataFrame(history.history)
    epoch_data.insert(0, "epoch", np.arange(1, len(epoch_data) + 1))
    epoch_data.to_csv(ABL_REPORTS / f"{trial}_history.csv", index=False)
    result = {
        "status": "ABLATION_TRIAL_PASS",
        "scope": "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ",
        "trial": trial, "config": TRIALS[trial],
        "control": "T3 selected: Word2Vec-initialized trainable Embedding + GRU64 + AdditiveAttention + Dense16 + Dropout0.20",
        "difference_from_control": (
            "Replace attention pooling with the final GRU state; keep Word2Vec initialization"
            if trial == "A1_no_attention" else
            "Replace TRAIN-only Word2Vec initialization with seeded random uniform; keep attention"
        ),
        "fixed_training": {
            "seed": SEED, "batch_size": stage3.BATCH_SIZE,
            "max_epochs": stage3.MAX_EPOCHS, "patience": stage3.PATIENCE,
            "learning_rate": stage3.LEARNING_RATE, "threshold": stage3.THRESHOLD,
            "trainable_embedding": True,
        },
        "train_count": len(y_train), "validation_count": len(y_val),
        "parameters_total": int(model.count_params()),
        "epochs_run": int(len(epoch_data)),
        "best_epoch_by_val_loss": int(np.argmin(epoch_data["val_loss"].to_numpy()) + 1),
        "best_val_loss": round(float(epoch_data["val_loss"].min()), 6),
        "validation_metrics": metrics,
        "confusion_matrix_validation_labels_0ham_1spam": matrix.tolist(),
        "validation_errors": int(np.sum(y_val != predicted)),
        "stage4_input_signature": signature,
        "classifier_sha256": sha256_file(saved),
        "model_reload_matches": True,
        "test_status": "LOCKED_NOT_EVALUATED",
    }
    report_path(trial).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{trial} PASS | F1-spam={metrics['f1_spam']:.6f} | CM={matrix.tolist()}", flush=True)
    print("TEST TIDAK DIBACA.", flush=True)


def finalize() -> None:
    stage3, stage2, selection, signature, _, _, x_val, y_val, _ = validated_inputs()
    trials = [valid_saved_trial(name, signature) for name in TRIALS]
    full = {
        "trial": "T3_full", "config": {"attention": True, "initialization": "train_only_word2vec"},
        "parameters_total": selection["winner_parameters_total"],
        "validation_metrics": selection["winner_validation_metrics"],
        "source": "STAGE3_T3_REUSED_NOT_RETRAINED",
    }
    rows = [full] + trials
    for row in trials:
        mat = np.asarray(row["confusion_matrix_validation_labels_0ham_1spam"], dtype=int)
        if mat.shape != (2, 2) or not np.array_equal(mat.sum(axis=1), np.bincount(y_val, minlength=2)):
            raise RuntimeError(f"Confusion matrix {row['trial']} tidak sesuai validation.")

    # Baseline T3 re-evaluation protects against stale or mismatched model files.
    tf = configure_tf()
    selected = tf.keras.models.load_model(SELECTED_MODEL, compile=False)
    full_prob = selected.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    actual = stage2.classification_metrics(y_val, (full_prob >= stage3.THRESHOLD).astype(np.int32), full_prob)
    if any(abs(actual[k] - v) > 2e-5 for k, v in full["validation_metrics"].items()):
        raise RuntimeError("Model T3 tersimpan tidak lagi sesuai dengan ringkasan Tahap 3.")

    flat = []
    for item in rows:
        flat.append({
            "variant": item["trial"],
            "word2vec_init": item["config"]["initialization"] == "train_only_word2vec",
            "attention": item["config"]["attention"],
            "parameters_total": item["parameters_total"],
            **item["validation_metrics"],
            "delta_f1_vs_full": round(
                item["validation_metrics"]["f1_spam"] - full["validation_metrics"]["f1_spam"], 6
            ),
            "delta_pr_auc_vs_full": round(
                item["validation_metrics"]["pr_auc_average_precision"] - full["validation_metrics"]["pr_auc_average_precision"], 6
            ),
        })
    frame = pd.DataFrame(flat)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(REPORTS / "uas_ablation_comparison.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4))
    categories = ["T3 Full", "No Attention", "No Word2Vec Init"]
    for ax, field, title in zip(
        axes, ("f1_spam", "pr_auc_average_precision"),
        ("F1 spam", "Average precision / PR-AUC"),
    ):
        bars = ax.bar(categories, frame[field], color=["#1a7799", "#94a7b6", "#94a7b6"])
        ax.set_ylim(0, 1.08)
        ax.set_title(title + " — validation")
        ax.tick_params(axis="x", labelrotation=14)
        for bar, value in zip(bars, frame[field]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                    f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ablation_gru_validation.png", dpi=160)
    plt.close(fig)

    summary = {
        "status": "UAS_ABLATION_VALIDATION_ONLY_PASS",
        "scope": "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ",
        "control_trial": "T3", "control_config": EXPECTED_CONFIG,
        "control_model_reused_not_retrained": True,
        "new_ablations_trained": 2,
        "validation_count": len(y_val),
        "comparison": flat,
        "stage4_input_signature": signature,
        "control_model_metrics_reverified": True,
        "test_status": "LOCKED_NOT_EVALUATED",
        "method_note": "Ablation compares one removed/replaced component at a time on the same fixed validation. Differences are conditional on this split and seed.",
        "interpretation_caution": "Attention is not a causal explanation; do not infer universal feature importance from this experiment.",
    }
    ABLATION_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 68)
    print("UAS TAHAP 4A — ABLATION VALIDATION ONLY")
    print("=" * 68)
    print(frame[["variant", "parameters_total", "f1_spam", "pr_auc_average_precision", "delta_f1_vs_full"]].to_string(index=False))
    print("Laporan:", ABLATION_SUMMARY.relative_to(ROOT))
    print("TEST: LOCKED_NOT_EVALUATED")
    print("UAS TAHAP 4A PASS — lanjut 10_interpret_validation.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation for selected T3 GRU64 model (validation only).")
    parser.add_argument("--trial", choices=tuple(TRIALS), help="Internal: fit one ablation.")
    parser.add_argument("--finalize", action="store_true", help="Rebuild summary from existing trials.")
    parser.add_argument("--force", action="store_true", help="Retrain both variants even if valid.")
    args = parser.parse_args()
    if args.trial and args.finalize:
        parser.error("--trial dan --finalize tidak dapat digabung.")
    if args.trial:
        train_one(args.trial)
        return
    if args.finalize:
        finalize()
        return
    print("=" * 68)
    print("UAS TAHAP 4A — ABLATION: WORD2VEC / ATTENTION (VALIDATION ONLY)")
    print("=" * 68)
    _, _, selection, signature, x_train, _, x_val, _, _ = validated_inputs()
    print(f"T3 model dipertahankan | TRAIN {len(x_train)} | VALIDATION {len(x_val)}")
    print(f"Baseline T3 F1-spam: {selection['winner_validation_metrics']['f1_spam']}")
    for trial in TRIALS:
        if not args.force and report_path(trial).is_file():
            try:
                valid_saved_trial(trial, signature)
            except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
                raise RuntimeError(f"Hasil {trial} tidak valid: {exc}. Gunakan --force untuk melatih ulang.") from exc
            print(f"{trial}: sudah PASS, training dilewati.", flush=True)
            continue
        print(f"{trial}: proses training terpisah dimulai...", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trial", trial], cwd=ROOT, check=True)
        valid_saved_trial(trial, signature)
        print(f"{trial}: selesai; memori TensorFlow anak proses dilepas.", flush=True)
    finalize()


if __name__ == "__main__":
    main()
