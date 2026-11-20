r"""UAS Tahap 3: four-point GRU+attention validation grid, CPU/RAM friendly.

Run from the project root:
    .\.venv\Scripts\python.exe scripts\08_tune_gru.py

T1 reuses the exact completed Stage-2 baseline (no retraining); T2-T4 run in
separate processes to release TensorFlow memory between configurations.
Only TRAIN and VALIDATION files are read. TEST remains locked.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
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
TUNE_MODELS = MODELS / "uas_tuning"
TUNE_REPORTS = REPORTS / "uas_tuning"
SCRIPT_STAGE2 = ROOT / "scripts" / "07_train_gru.py"
STAGE2_REPORT = REPORTS / "hasil_validasi_gru_attention.json"
BASELINE_CLASSIFIER = MODELS / "gru_attention_baseline.keras"
BASELINE_ATTENTION = MODELS / "gru_attention_reader.keras"

SEED = 42
THRESHOLD = 0.50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 15
PATIENCE = 3
DENSE_UNITS = 16

TRIALS = {
    "T1": {"gru_units": 32, "dropout": 0.20},  # Reuse Stage-2 model.
    "T2": {"gru_units": 32, "dropout": 0.40},
    "T3": {"gru_units": 64, "dropout": 0.20},
    "T4": {"gru_units": 64, "dropout": 0.40},
}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Arsip prasyarat belum ada: {path}")
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def load_stage2():
    if not SCRIPT_STAGE2.is_file():
        raise FileNotFoundError("scripts/07_train_gru.py tidak ditemukan (Tahap 2).")
    spec = importlib.util.spec_from_file_location("_uas_stage2_reuse", SCRIPT_STAGE2)
    if spec is None or spec.loader is None:
        raise RuntimeError("Tidak dapat mengimpor pemeriksa Tahap 2.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, value in (
        ("GRU_UNITS", 32), ("DENSE_UNITS", DENSE_UNITS),
        ("DROPOUT_RATE", 0.20), ("LEARNING_RATE", LEARNING_RATE),
        ("BATCH_SIZE", BATCH_SIZE), ("MAX_EPOCHS", MAX_EPOCHS),
        ("PATIENCE", PATIENCE), ("THRESHOLD", THRESHOLD), ("SEED", SEED),
    ):
        if getattr(module, name, object()) != value:
            raise RuntimeError(f"Konfigurasi Tahap 2 berubah: {name}. Jangan tuning data lama.")
    return module


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Laporan prasyarat tidak tersedia: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def input_signature(manifest: dict) -> dict:
    return {
        "train_csv_sha256": manifest["train_sha256"],
        "validation_csv_sha256": manifest["validation_sha256"],
        "sequences_train_sha256": sha256_file(ROOT / "data/processed/sequences_train.npz"),
        "sequences_validation_sha256": sha256_file(ROOT / "data/processed/sequences_validation.npz"),
        "word2vec_matrix_sha256": sha256_file(MODELS / "word2vec_embedding_matrix.npy"),
        "vocabulary_sha256": sha256_file(MODELS / "sequence_vocabulary.json"),
        "tokenizer_config_sha256": sha256_file(MODELS / "sequence_tokenizer_config.json"),
        "baseline_classifier_sha256": sha256_file(BASELINE_CLASSIFIER),
        "baseline_attention_sha256": sha256_file(BASELINE_ATTENTION),
        "baseline_report_sha256": sha256_file(STAGE2_REPORT),
    }


def validated_inputs():
    stage2 = load_stage2()
    manifest, x_train, y_train, x_val, y_val, embedding = stage2.load_and_verify_inputs()
    baseline = read_json(STAGE2_REPORT)
    if baseline.get("status") != "GRU_ATTENTION_TRAINED_VALIDATION_ONLY":
        raise RuntimeError("Tahap 2 belum PASS. Latih 07_train_gru.py terlebih dahulu.")
    if baseline.get("scope") != "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ":
        raise RuntimeError("Scope baseline bukan TRAIN/VALIDATION only.")
    if baseline.get("test_status") != "LOCKED_NOT_EVALUATED":
        raise RuntimeError("Dataset TEST dilaporkan pernah digunakan oleh Tahap 2.")
    if baseline.get("train_and_validation_source_sha256") != {
        "train": manifest["train_sha256"],
        "validation": manifest["validation_sha256"],
    }:
        raise RuntimeError("Checksum data Stage 2 tidak sama dengan Stage 1.")
    if (baseline.get("train_count"), baseline.get("validation_count"),
            baseline.get("sequence_length")) != (len(y_train), len(y_val), x_train.shape[1]):
        raise RuntimeError("Ukuran train/validation Stage 2 dan Stage 1 berbeda.")
    if baseline.get("embedding", {}).get("vocabulary_size") != embedding.shape[0]:
        raise RuntimeError("Ukuran vocabulary Stage 2 dan Stage 1 berbeda.")
    if baseline.get("embedding", {}).get("dimension") != embedding.shape[1]:
        raise RuntimeError("Dimensi Word2Vec Stage 2 dan Stage 1 berbeda.")
    if baseline.get("seed") != SEED or baseline.get("threshold") != THRESHOLD:
        raise RuntimeError("Seed atau threshold baseline telah berubah.")
    if int(baseline.get("parameters_total", -1)) != 230337:
        raise RuntimeError("Arsitektur baseline Tahap 2 berubah; butuh pemeriksaan manual.")
    signature = input_signature(manifest)
    return stage2, baseline, signature, x_train, y_train, x_val, y_val, embedding


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


def build_models(tf, embedding: np.ndarray, sequence_length: int, config: dict):
    units = config["gru_units"]
    rate = config["dropout"]
    token_ids = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="token_ids")
    vectors = tf.keras.layers.Embedding(
        input_dim=embedding.shape[0], output_dim=embedding.shape[1],
        weights=[embedding], mask_zero=True, trainable=True,
        name="word2vec_embedding",
    )(token_ids)
    sequence, last_state = tf.keras.layers.GRU(
        units, return_sequences=True, return_state=True, name="gru_encoder",
    )(vectors)
    query = tf.keras.layers.Reshape((1, units), name="attention_query")(last_state)
    context, attention_weights = tf.keras.layers.AdditiveAttention(
        use_scale=True, dropout=0.0, name="token_attention",
    )([query, sequence], return_attention_scores=True)
    context = tf.keras.layers.Flatten(name="attention_context")(context)
    context = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="hidden_16")(context)
    context = tf.keras.layers.Dropout(rate, name=f"dropout_{round(rate * 100):03d}")(context)
    probability = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(context)
    classifier = tf.keras.Model(token_ids, probability, name="sms_spam_gru_attention_tuning")
    attention_reader = tf.keras.Model(token_ids, attention_weights, name="sms_spam_attention_reader")
    return classifier, attention_reader


def trial_model_paths(trial: str):
    return (
        TUNE_MODELS / f"{trial}_classifier.keras",
        TUNE_MODELS / f"{trial}_attention_reader.keras",
    )


def trial_result_path(trial: str) -> Path:
    return TUNE_REPORTS / f"{trial}_validation.json"


def validate_saved_trial(trial: str, signature: dict) -> dict:
    result = read_json(trial_result_path(trial))
    classifier, attention = trial_model_paths(trial)
    if (result.get("status") != "GRU_TUNING_TRIAL_PASS"
            or result.get("trial") != trial or result.get("config") != TRIALS[trial]
            or result.get("input_signature") != signature
            or result.get("test_status") != "LOCKED_NOT_EVALUATED"):
        raise RuntimeError(f"Artefak {trial} tidak cocok dengan input atau konfigurasi saat ini.")
    if (sha256_file(classifier) != result.get("classifier_sha256")
            or sha256_file(attention) != result.get("attention_reader_sha256")):
        raise RuntimeError(f"Berkas model {trial} berubah atau rusak sejak pelatihan.")
    if not isinstance(result.get("validation_metrics"), dict):
        raise RuntimeError(f"Laporan metrik {trial} tidak lengkap.")
    return result


def run_trial(trial: str) -> None:
    if trial not in ("T2", "T3", "T4"):
        raise ValueError("T1 adalah baseline Tahap 2; tidak perlu dilatih ulang.")
    stage2, _, signature, x_train, y_train, x_val, y_val, embedding = validated_inputs()
    print(f"\n{'=' * 70}\nUAS TAHAP 3 - TRAINING {trial}: {TRIALS[trial]}\n{'=' * 70}", flush=True)
    tf = configure_tf()
    classifier, attention_reader = build_models(tf, embedding, x_train.shape[1], TRIALS[trial])
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    classifier.summary()
    history = classifier.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
        )],
    )
    proba = classifier.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    if not np.isfinite(proba).all():
        raise RuntimeError("Model menghasilkan probabilitas NaN/Infinity.")
    predictions = (proba >= THRESHOLD).astype(np.int32)
    metrics = stage2.classification_metrics(y_val, predictions, proba)
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_val, predictions, labels=[0, 1])
    if int(confusion.sum()) != len(y_val):
        raise RuntimeError("Ukuran confusion matrix tidak sesuai.")

    probe_x = x_val[: min(32, len(x_val))]
    att = attention_reader.predict(probe_x, verbose=0)
    masked_weights = att[:, 0, :][probe_x == 0]
    max_padding = float(np.abs(masked_weights).max(initial=0.0))
    if (att.shape != (len(probe_x), 1, x_val.shape[1])
            or not np.allclose(att.sum(axis=-1), 1, atol=1e-4)
            or max_padding > 1e-4):
        raise RuntimeError(f"Attention atau masking {trial} tidak valid.")

    TUNE_MODELS.mkdir(parents=True, exist_ok=True)
    TUNE_REPORTS.mkdir(parents=True, exist_ok=True)
    model_path, reader_path = trial_model_paths(trial)
    classifier.save(model_path)
    attention_reader.save(reader_path)
    loaded = tf.keras.models.load_model(model_path, compile=False)
    loaded_reader = tf.keras.models.load_model(reader_path, compile=False)
    if not np.allclose(
        loaded.predict(probe_x, verbose=0).reshape(-1),
        classifier.predict(probe_x, verbose=0).reshape(-1), atol=1e-6,
    ):
        raise RuntimeError(f"Model {trial} gagal reload.")
    if not np.allclose(loaded_reader.predict(probe_x, verbose=0), att, atol=1e-6):
        raise RuntimeError(f"Pembaca attention {trial} gagal reload.")

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(TUNE_REPORTS / f"{trial}_training_history.csv", index=False)
    result = {
        "status": "GRU_TUNING_TRIAL_PASS",
        "scope": "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ",
        "trial": trial,
        "config": TRIALS[trial],
        "fixed_hyperparameters": {
            "dense_units": DENSE_UNITS, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
            "learning_rate": LEARNING_RATE, "threshold": THRESHOLD, "seed": SEED,
            "trainable_word2vec": True, "attention": "AdditiveAttention",
        },
        "train_count": len(y_train), "validation_count": len(y_val),
        "parameters_total": int(classifier.count_params()),
        "epochs_run": len(history_df),
        "best_epoch_by_val_loss": int(np.argmin(history_df["val_loss"].to_numpy()) + 1),
        "best_val_loss": round(float(history_df["val_loss"].min()), 6),
        "validation_metrics": metrics,
        "confusion_matrix_validation_labels_0ham_1spam": confusion.tolist(),
        "validation_errors": int(np.sum(y_val != predictions)),
        "attention_checks": {
            "checked_samples": len(probe_x), "weights_sum_to_one": True,
            "max_padding_weight": round(max_padding, 8), "reload_matches": True,
        },
        "model_reload_matches": True,
        "input_signature": signature,
        "classifier_sha256": sha256_file(model_path),
        "attention_reader_sha256": sha256_file(reader_path),
        "test_status": "LOCKED_NOT_EVALUATED",
    }
    trial_result_path(trial).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{trial} PASS | F1-spam={metrics['f1_spam']:.6f} | CM={confusion.tolist()}", flush=True)
    print("TEST TETAP TERKUNCI.", flush=True)


def ranking_key(record: dict) -> tuple:
    metric = record["validation_metrics"]
    return (
        -float(metric["f1_spam"]),
        -float(metric["recall_spam"]),
        -float(metric["pr_auc_average_precision"]),
        int(record["parameters_total"]),
        record["trial"],
    )


def finalize() -> None:
    stage2, baseline, signature, _, _, x_val, y_val, _ = validated_inputs()
    if not STAGE2_REPORT.is_file():
        raise FileNotFoundError("Laporan baseline GRU Stage 2 tidak ditemukan.")
    base_row = {
        "trial": "T1", "config": TRIALS["T1"],
        "source": "STAGE_2_BASELINE_REUSED",
        "epochs_run": baseline["epochs_run"],
        "best_epoch_by_val_loss": baseline["best_epoch_by_val_loss"],
        "best_val_loss": baseline["best_val_loss"],
        "parameters_total": baseline["parameters_total"],
        "validation_metrics": baseline["validation_gru_attention"],
        "confusion_matrix_validation_labels_0ham_1spam": baseline[
            "confusion_matrix_validation_labels_0ham_1spam"
        ],
    }
    records = [base_row] + [validate_saved_trial(t, signature) for t in ("T2", "T3", "T4")]
    # Verify reported confusion matrices without touching TEST.
    for row in records:
        cm = np.array(row["confusion_matrix_validation_labels_0ham_1spam"], dtype=int)
        if cm.shape != (2, 2) or cm.min() < 0 or cm.sum() != len(y_val):
            raise RuntimeError(f"Confusion matrix {row['trial']} tidak sesuai validation.")
        if (cm.sum(axis=1) != np.bincount(y_val, minlength=2)).any():
            raise RuntimeError(f"Distribusi label {row['trial']} berbeda dari validation.")

    ranked = sorted(records, key=ranking_key)
    winner = ranked[0]
    winner_id = winner["trial"]
    winner_classifier, winner_attention = (
        (BASELINE_CLASSIFIER, BASELINE_ATTENTION)
        if winner_id == "T1" else trial_model_paths(winner_id)
    )

    # Verify the selected model produces the stated metrics and attention mask.
    tf = configure_tf()
    model = tf.keras.models.load_model(winner_classifier, compile=False)
    reader = tf.keras.models.load_model(winner_attention, compile=False)
    probability = model.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    actual = stage2.classification_metrics(y_val, (probability >= THRESHOLD).astype(np.int32), probability)
    for name, reported in winner["validation_metrics"].items():
        if abs(actual[name] - reported) > 2e-5:
            raise RuntimeError(f"Metrik {name} model terpilih tidak sama dengan laporan ({reported} vs {actual[name]}).")
    probe = x_val[: min(32, len(x_val))]
    scores = reader.predict(probe, verbose=0)
    if (scores.shape != (len(probe), 1, x_val.shape[1])
            or not np.allclose(scores.sum(axis=-1), 1, atol=1e-4)
            or np.abs(scores[:, 0, :][probe == 0]).max(initial=0) > 1e-4):
        raise RuntimeError("Attention model terpilih gagal validasi masking.")

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    final_classifier = MODELS / "gru_attention_tuned.keras"
    final_reader = MODELS / "gru_attention_tuned_reader.keras"
    shutil.copy2(winner_classifier, final_classifier)
    shutil.copy2(winner_attention, final_reader)

    flat_records = []
    for row in records:
        flat_records.append({
            "trial": row["trial"], "gru_units": row["config"]["gru_units"],
            "dropout": row["config"]["dropout"],
            "source": row.get("source", "TUNING_NEW_TRAINING"),
            "parameters_total": row["parameters_total"],
            "epochs_run": row["epochs_run"],
            "best_epoch": row["best_epoch_by_val_loss"],
            "best_val_loss": row["best_val_loss"],
            **row["validation_metrics"],
            "selected": row["trial"] == winner_id,
        })
    frame = pd.DataFrame(flat_records)
    frame.to_csv(REPORTS / "uas_tuning_grid_results.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, name, label in zip(axes, ("f1_spam", "precision_spam", "recall_spam"),
                               ("F1 spam", "Precision spam", "Recall spam")):
        bars = ax.bar(frame["trial"], frame[name], color=[
            "#167da2" if t == winner_id else "#9bafbb" for t in frame["trial"]
        ])
        ax.set_ylim(0, 1.07)
        ax.set_title(label)
        ax.set_ylabel("Validation score")
        for bar, value in zip(bars, frame[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + .015,
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("UAS GRU + Attention: four configurations (validation only)")
    fig.tight_layout()
    fig.savefig(FIGURES / "tuning_gru_validation.png", dpi=160)
    plt.close(fig)

    summary = {
        "status": "UAS_TUNING_VALIDATION_ONLY_PASS",
        "scope": "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ",
        "baseline_stage2_reused_without_retraining": True,
        "experiment_count": 4,
        "new_models_trained": 3,
        "search_strategy": "fixed small grid",
        "selection_rule": [
            "highest validation F1-spam", "highest validation recall-spam",
            "highest validation average precision (PR-AUC)",
            "fewest trainable parameters", "lexicographically smallest trial id",
        ],
        "winner_trial": winner_id,
        "winner_config": winner["config"],
        "winner_validation_metrics": winner["validation_metrics"],
        "winner_parameters_total": winner["parameters_total"],
        "baseline_validation_metrics": baseline["validation_gru_attention"],
        "trials": flat_records,
        "input_signature": signature,
        "selected_classifier": str(final_classifier.relative_to(ROOT)).replace("\\", "/"),
        "selected_attention_reader": str(final_reader.relative_to(ROOT)).replace("\\", "/"),
        "selected_classifier_sha256": sha256_file(final_classifier),
        "selected_attention_reader_sha256": sha256_file(final_reader),
        "selected_model_reload_metrics_match": True,
        "selected_attention_mask_pass": True,
        "test_status": "LOCKED_NOT_EVALUATED",
        "caution": "Validation dipakai untuk pemilihan model; TEST belum digunakan. Perbedaan kecil metrik tidak menjamin peningkatan generalisasi.",
    }
    (REPORTS / "uas_tuning_selection.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 70)
    print("UAS TAHAP 3 - RINGKASAN VALIDATION (TEST TIDAK DIBACA)")
    print("=" * 70)
    print(frame[["trial", "gru_units", "dropout", "f1_spam", "recall_spam", "pr_auc_average_precision", "selected"]].to_string(index=False))
    print(f"MODEL TERPILIH: {winner_id} {winner['config']} | F1-spam: {winner['validation_metrics']['f1_spam']:.6f}")
    print("Laporan:", (REPORTS / "uas_tuning_selection.json").relative_to(ROOT))
    print("Model terpilih:", final_classifier.relative_to(ROOT))
    print("TEST: LOCKED_NOT_EVALUATED")
    print("UAS TAHAP 3 PASS - siap ablation dan analisis kesalahan VALIDATION.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tuning GRU+Attention validation-only (T1 reused from Stage 2).")
    parser.add_argument("--trial", choices=("T2", "T3", "T4"),
                        help="Mode internal: melatih satu konfigurasi.")
    parser.add_argument("--finalize", action="store_true",
                        help="Menyusun peringkat dari hasil eksperimen yang sudah tersedia.")
    parser.add_argument("--force", action="store_true",
                        help="Melatih ulang T2-T4 meskipun hasil valid sudah tersedia.")
    args = parser.parse_args()
    if args.trial and args.finalize:
        parser.error("--trial dan --finalize tidak dapat digabungkan.")
    if args.trial:
        run_trial(args.trial)
        return
    if args.finalize:
        finalize()
        return

    print("=" * 70)
    print("UAS TAHAP 3 - GRID SEARCH GRU + ATTENTION (VALIDATION ONLY)")
    print("=" * 70)
    print("T1: GRU32/dropout0.20 -> baseline Tahap 2, tidak dilatih ulang")
    print("T2: GRU32/dropout0.40 | T3: GRU64/dropout0.20 | T4: GRU64/dropout0.40")
    _, baseline, signature, x_train, _, x_val, _, _ = validated_inputs()
    print(f"TRAIN {len(x_train):,} | VALIDATION {len(x_val):,} | TEST tidak dibaca")
    print("T1 baseline F1-spam:", baseline["validation_gru_attention"]["f1_spam"])
    for trial in ("T2", "T3", "T4"):
        if not args.force and trial_result_path(trial).exists():
            try:
                validate_saved_trial(trial, signature)
            except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:
                raise RuntimeError(
                    f"{trial} lama tidak valid: {exc}. Jalankan kembali dengan --force jika ingin regenerasi."
                ) from exc
            print(f"{trial}: artefak VALID, dilewati (melanjutkan hasil sebelumnya).", flush=True)
            continue
        print(f"{trial}: memulai proses training terpisah ...", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--trial", trial],
                       cwd=ROOT, check=True)
        validate_saved_trial(trial, signature)
        print(f"{trial}: pelatihan selesai, memori TensorFlow dilepas.", flush=True)
    finalize()


if __name__ == "__main__":
    main()
