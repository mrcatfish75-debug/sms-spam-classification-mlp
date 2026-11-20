"""UAS Tahap 2: Word2Vec-initialized GRU + AdditiveAttention (VALIDATION only).

Run from project root on Python 3.13 (Windows CPU also supported):
    .\\.venv\\Scripts\\python.exe scripts\\07_train_gru.py

Prerequisite: 06_prepare_sequences.py must have completed successfully.
Never reads data/splits/sms_test.csv. Do not tune to TEST outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"
MANIFEST = REPORTS / "uas_sequence_word2vec_manifest.json"

SEED = 42
GRU_UNITS = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.20
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 15
PATIENCE = 3
THRESHOLD = 0.50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sequences(part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = PROCESSED / f"sequences_{part}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Berkas sekuensial belum ditemukan: {path}")
    with np.load(path, allow_pickle=False) as contents:
        if set(contents.files) != {"x", "y", "source_row"}:
            raise ValueError(f"{path.name}: field wajib x, y, source_row tidak cocok.")
        x = contents["x"].astype(np.int32, copy=False)
        y = contents["y"].astype(np.int32, copy=False)
        source_ids = contents["source_row"].astype(np.int64, copy=False)
    if x.ndim != 2 or y.ndim != 1 or source_ids.ndim != 1:
        raise ValueError(f"{part}: dimensi x/y/source_row tidak valid.")
    if not (len(x) == len(y) == len(source_ids)):
        raise ValueError(f"{part}: ukuran x/y/source_row tidak sama.")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) < 2:
        raise ValueError(f"{part}: label harus biner dan memuat ham serta spam.")
    if np.any(x < 0) or not np.all(x[:, 0] > 0):
        raise ValueError(f"{part}: ID token negatif atau baris dimulai PAD.")
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError(f"{part}: source_row duplikat.")
    return x, y, source_ids


def load_and_verify_inputs() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not MANIFEST.is_file():
        raise FileNotFoundError("Manifest Word2Vec tidak ditemukan. Jalankan Tahap 1 dahulu.")
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if m.get("status") != "SEQUENCE_WORD2VEC_PASS":
        raise RuntimeError("Tahap 1 belum PASS; batalkan pelatihan GRU.")
    if m.get("test_status") != "LOCKED_NOT_READ":
        raise RuntimeError("Status test pada manifest Tahap 1 tidak terkunci.")
    if m.get("scope") != "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ":
        raise RuntimeError("Scope Tahap 1 tidak sesuai protokol.")

    for part in ("train", "validation"):
        split = SPLITS / f"sms_{part}.csv"
        if not split.is_file():
            raise FileNotFoundError(f"Split UTS {part} tidak ditemukan: {split}")
        digest = sha256_file(split)
        if digest != m.get(f"{part}_sha256"):
            raise RuntimeError(f"Berkas {split.name} berbeda dari saat Word2Vec dibuat.")

    x_train, y_train, ids_train = read_sequences("train")
    x_val, y_val, ids_val = read_sequences("validation")
    if len(set(ids_train).intersection(set(ids_val))):
        raise RuntimeError("LEAKAGE: source_row train dan validation beririsan.")

    # Cross-check the label/row alignment between the original UTS split and NPZ.
    # These are TRAIN and VALIDATION CSV only; TEST remains untouched.
    for part, y, source_ids in (
        ("train", y_train, ids_train),
        ("validation", y_val, ids_val),
    ):
        records = pd.read_csv(
            SPLITS / f"sms_{part}.csv",
            encoding="utf-8-sig",
            keep_default_na=False,
            usecols=["source_row", "label"],
        )
        expected_ids = records["source_row"].to_numpy(dtype=np.int64)
        expected_y = (records["label"].astype(str).str.strip().str.lower() == "spam").to_numpy(dtype=np.int32)
        if not np.array_equal(source_ids, expected_ids) or not np.array_equal(y, expected_y):
            raise RuntimeError(f"Sekuensial {part} tidak sebaris dengan split UTS aslinya.")

    config_path = MODELS / "sequence_tokenizer_config.json"
    vocab_path = MODELS / "sequence_vocabulary.json"
    embedding_path = MODELS / "word2vec_embedding_matrix.npy"
    for path in (config_path, vocab_path, embedding_path):
        if not path.is_file():
            raise FileNotFoundError(f"Artefak Tahap 1 belum tersedia: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    embedding = np.load(embedding_path, allow_pickle=False).astype(np.float32, copy=False)

    expected_length = int(m["tokenization"]["sequence_length"])
    expected_dim = int(m["tokenization"]["embedding_dim"])
    expected_vocab = int(m["tokenization"]["vocabulary_size"])
    if (
        tuple(x_train.shape) != (int(m["train_count"]), expected_length)
        or tuple(x_val.shape) != (int(m["validation_count"]), expected_length)
        or tuple(x_train.shape) != tuple(m["sequence_train_shape"])
        or tuple(x_val.shape) != tuple(m["sequence_validation_shape"])
    ):
        raise RuntimeError("Bentuk sekuensial tidak cocok dengan manifest Tahap 1.")
    if embedding.shape != (expected_vocab, expected_dim):
        raise RuntimeError("Bentuk embedding tidak cocok dengan manifest Tahap 1.")
    if len(vocab) != expected_vocab or vocab[:2] != ["[PAD]", "[UNK]"]:
        raise RuntimeError("Vocabulary tidak cocok dengan embedding PAD/UNK.")
    if not np.all(embedding[0] == 0):
        raise RuntimeError("Baris PAD pada embedding seharusnya bernilai nol.")
    if not np.isfinite(embedding).all():
        raise RuntimeError("Embedding mengandung NaN atau infinity.")
    if int(x_train.max()) >= expected_vocab or int(x_val.max()) >= expected_vocab:
        raise RuntimeError("Ada token ID di luar vocabulary.")
    if config.get("vocabulary_size") != expected_vocab or config.get("sequence_length") != expected_length:
        raise RuntimeError("Konfigurasi tokenizer tidak konsisten.")
    if config != m.get("tokenization"):
        raise RuntimeError("Konfigurasi tokenizer berbeda dari manifest Tahap 1.")

    return m, x_train, y_train, x_val, y_val, embedding


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_spam": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall_spam": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_spam": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "pr_auc_average_precision": round(float(average_precision_score(y_true, proba)), 6),
    }


def build_models(tf, embedding: np.ndarray, sequence_length: int):
    token_ids = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="token_ids")
    token_vectors = tf.keras.layers.Embedding(
        input_dim=embedding.shape[0],
        output_dim=embedding.shape[1],
        weights=[embedding],
        mask_zero=True,
        trainable=True,
        name="word2vec_embedding",
    )(token_ids)
    sequence_output, final_state = tf.keras.layers.GRU(
        GRU_UNITS,
        return_sequences=True,
        return_state=True,
        name="gru_encoder",
    )(token_vectors)
    query = tf.keras.layers.Reshape((1, GRU_UNITS), name="attention_query")(final_state)
    context, attention_scores = tf.keras.layers.AdditiveAttention(
        name="token_attention",
        use_scale=True,
        dropout=0.0,
    )([query, sequence_output], return_attention_scores=True)
    context = tf.keras.layers.Flatten(name="attention_context")(context)
    context = tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="hidden_16")(context)
    context = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout_020")(context)
    spam_probability = tf.keras.layers.Dense(1, activation="sigmoid", name="spam_probability")(context)

    classifier = tf.keras.Model(token_ids, spam_probability, name="sms_spam_gru_attention_uas")
    attention_reader = tf.keras.Model(token_ids, attention_scores, name="sms_spam_attention_reader")
    return classifier, attention_reader


def make_figures(history: pd.DataFrame, confusion: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for column, title, y_axis, axis in (
        ("loss", "Loss", "Binary crossentropy", axes[0]),
        ("accuracy", "Accuracy", "Accuracy", axes[1]),
    ):
        axis.plot(history["epoch"], history[column], label="Train")
        axis.plot(history["epoch"], history[f"val_{column}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=y_axis, title=title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "kurva_training_gru_attention.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1], ["Ham", "Spam"])
    ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix GRU - VALIDATION")
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix_gru_validation.png", dpi=160)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("UAS TAHAP 2 - TRAINING GRU + ATTENTION (VALIDATION ONLY)")
    print("=" * 72)
    manifest, x_train, y_train, x_val, y_val, embedding = load_and_verify_inputs()
    print(f"TRAIN: {x_train.shape} | VALIDATION: {x_val.shape}")
    print(f"Word2Vec embedding: {embedding.shape} | TEST tidak dibaca")

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    tf.keras.backend.clear_session()

    model, attention_reader = build_models(tf, embedding, x_train.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True,
    )]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    proba = model.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    predictions = (proba >= THRESHOLD).astype(np.int32)
    metrics = classification_metrics(y_val, predictions, proba)
    confusion = confusion_matrix(y_val, predictions, labels=[0, 1])
    if int(confusion.sum()) != len(y_val):
        raise RuntimeError("Confusion matrix tidak mencakup seluruh validation.")

    # Attention reader shares the trained weights with the classifier.
    probe_x = x_val[: min(32, len(x_val))]
    attention = attention_reader.predict(probe_x, verbose=0)
    if attention.shape != (len(probe_x), 1, x_val.shape[1]):
        raise RuntimeError(f"Bentuk attention tidak sesuai: {attention.shape}")
    totals = attention.sum(axis=-1)
    max_pad_attention = float(np.abs(attention[:, 0, :][probe_x == 0]).max(initial=0.0))
    if not np.allclose(totals, 1.0, atol=1e-4) or max_pad_attention > 1e-4:
        raise RuntimeError(
            "Masking attention tidak valid: bobot pada PAD tidak nol atau jumlahnya tidak 1."
        )

    for folder in (MODELS, REPORTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    model_path = MODELS / "gru_attention_baseline.keras"
    attention_path = MODELS / "gru_attention_reader.keras"
    model.save(model_path)
    attention_reader.save(attention_path)

    # Round-trip load catches incompatible/nonserializable Keras layers early.
    loaded = tf.keras.models.load_model(model_path, compile=False)
    replay = loaded.predict(probe_x, verbose=0).reshape(-1)
    reference = model.predict(probe_x, verbose=0).reshape(-1)
    if not np.allclose(replay, reference, atol=1e-6):
        raise RuntimeError("Model tersimpan tidak menghasilkan prediksi yang sama setelah reload.")
    loaded_attention = tf.keras.models.load_model(attention_path, compile=False)
    replay_scores = loaded_attention.predict(probe_x, verbose=0)
    if not np.allclose(replay_scores, attention, atol=1e-6):
        raise RuntimeError("Attention reader berubah setelah disimpan dan dimuat kembali.")

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(REPORTS / "gru_attention_training_history.csv", index=False)
    report_text = classification_report(
        y_val, predictions, labels=[0, 1], target_names=["ham", "spam"], zero_division=0,
    )
    (REPORTS / "classification_report_gru_validation.txt").write_text(
        report_text, encoding="utf-8"
    )
    best_epoch = int(np.argmin(history_df["val_loss"].to_numpy()) + 1)
    result = {
        "status": "GRU_ATTENTION_TRAINED_VALIDATION_ONLY",
        "scope": "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ",
        "seed": SEED,
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "model": "Word2Vec-initialized trainable Embedding -> GRU32 -> AdditiveAttention -> Dense16 ReLU -> Dropout0.20 -> Sigmoid",
        "embedding": {
            "vocabulary_size": int(embedding.shape[0]),
            "dimension": int(embedding.shape[1]),
            "trainable": True,
            "source": "Word2Vec trained from TRAIN only in 06_prepare_sequences.py",
        },
        "train_count": int(len(x_train)),
        "validation_count": int(len(x_val)),
        "sequence_length": int(x_train.shape[1]),
        "parameters_total": int(model.count_params()),
        "optimizer": f"Adam(learning_rate={LEARNING_RATE})",
        "loss": "binary_crossentropy",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "epochs_run": int(len(history_df)),
        "best_epoch_by_val_loss": best_epoch,
        "best_val_loss": round(float(history_df["val_loss"].min()), 6),
        "early_stopping_patience": PATIENCE,
        "threshold": THRESHOLD,
        "train_class_counts": {"ham": int(np.sum(y_train == 0)), "spam": int(np.sum(y_train == 1))},
        "validation_class_counts": {"ham": int(np.sum(y_val == 0)), "spam": int(np.sum(y_val == 1))},
        "validation_gru_attention": metrics,
        "confusion_matrix_validation_labels_0ham_1spam": confusion.tolist(),
        "validation_errors": int((y_val != predictions).sum()),
        "attention_probe": {
            "file": str(attention_path.relative_to(ROOT)).replace("\\", "/"),
            "checked_samples": len(probe_x),
            "scores_sum_to_one": True,
            "max_attention_on_padding": round(max_pad_attention, 8),
            "reload_matches": True,
        },
        "model_reload_matches": True,
        "train_and_validation_source_sha256": {
            "train": manifest["train_sha256"],
            "validation": manifest["validation_sha256"],
        },
        "test_status": "LOCKED_NOT_EVALUATED",
        "note": "Semua hasil GRU di file ini VALIDATION, bukan TEST. Jangan pilih model menggunakan TEST.",
    }
    (REPORTS / "hasil_validasi_gru_attention.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(history_df, confusion)

    print("\nHASIL GRU + ATTENTION (VALIDATION - BUKAN TEST)")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"Epoch dijalankan: {len(history_df)} | Epoch terbaik: {best_epoch}")
    print("Confusion matrix (aktual ham/spam x prediksi ham/spam):", confusion.tolist())
    print(report_text)
    print("Model tersimpan:", model_path.relative_to(ROOT))
    print("Pembaca attention tersimpan:", attention_path.relative_to(ROOT))
    print("TRAIN/VALIDATION ONLY; TEST TETAP TERKUNCI.")
    print("UAS TAHAP 2 PASS - siap lanjut hyperparameter tuning.")


if __name__ == "__main__":
    main()
