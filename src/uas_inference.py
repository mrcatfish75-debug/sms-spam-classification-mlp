"""Read-only inference with the *frozen* UTS MLP and UAS T3 GRU models.

This module deliberately does not access data/raw, data/processed, data/splits,
or any held-out evaluation file. It never trains, fits, evaluates a dataset,
changes the threshold, or writes user-provided SMS to disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FREEZE = ROOT / "outputs" / "reports" / "uas_final_protocol_lock.json"
THRESHOLD = 0.5
MAX_MESSAGE_CHARS = 2000

FROZEN_INPUTS = (
    "models/mlp_baseline.keras",
    "models/tfidf_vectorizer.joblib",
    "models/gru_attention_tuned.keras",
    "models/gru_attention_tuned_reader.keras",
    "models/sequence_vocabulary.json",
    "models/sequence_tokenizer_config.json",
)


@dataclass(frozen=True)
class TextEncoding:
    sequence: np.ndarray
    tokens: tuple[str, ...]
    unknown_count: int
    truncated: bool

    @property
    def unknown_fraction(self) -> float:
        return self.unknown_count / len(self.tokens) if self.tokens else 0.0


@dataclass(frozen=True)
class FrozenArtifacts:
    vocabulary: tuple[str, ...]
    word_to_id: dict[str, int]
    token_pattern: str
    sequence_length: int
    freeze_status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def frozen_artifacts() -> FrozenArtifacts:
    """Validate signatures without opening a dataset or loading TensorFlow."""
    if not FREEZE.is_file():
        raise FileNotFoundError(
            "Protokol freeze UAS tidak ditemukan: outputs/reports/uas_final_protocol_lock.json"
        )
    lock = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (
        lock.get("status") != "UAS_FINAL_EVALUATION_PROTOCOL_FROZEN"
        or lock.get("selected_trial") != "T3"
        or lock.get("model_uas") != "models/gru_attention_tuned.keras"
        or lock.get("model_uts") != "models/mlp_baseline.keras"
        or lock.get("threshold") != THRESHOLD
    ):
        raise RuntimeError("Konfigurasi model/threshold tidak sesuai protokol terkunci.")

    registered_hashes = lock.get("input_sha256", {})
    for relative_path in FROZEN_INPUTS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Artefak model tidak ditemukan: {relative_path}")
        expected = registered_hashes.get(relative_path)
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(
                f"Checksum model tidak sesuai protokol freeze: {relative_path}. "
                "Jangan mengganti atau melatih ulang model."
            )

    config = json.loads((MODELS / "sequence_tokenizer_config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((MODELS / "sequence_vocabulary.json").read_text(encoding="utf-8"))
    if (
        config.get("standardization") != "NFKC + casefold"
        or config.get("token_pattern") != r"(?u)\b\w+\b|[^\w\s]"
        or config.get("sequence_length") != 60
        or config.get("vocabulary_size") != 3443
        or config.get("unknown_token_index") != 1
        or config.get("embedding_dim") != 64
        or not isinstance(vocabulary, list)
        or len(vocabulary) != 3443
        or vocabulary[:2] != ["[PAD]", "[UNK]"]
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise RuntimeError("Konfigurasi tokenizer tidak cocok dengan UAS Tahap 1.")

    return FrozenArtifacts(
        vocabulary=tuple(vocabulary),
        word_to_id={token: index for index, token in enumerate(vocabulary)},
        token_pattern=config["token_pattern"],
        sequence_length=config["sequence_length"],
        freeze_status=lock["status"],
    )


def validate_message(message: str) -> str:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Masukkan teks SMS terlebih dahulu.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Panjang maksimal {MAX_MESSAGE_CHARS} karakter untuk satu pesan.")
    return message


def encode_sms(message: str) -> TextEncoding:
    """Exactly mirrors Stage 1: NFKC, casefold, regex, UNK=1, post-padding."""
    text = validate_message(message)
    artifact = frozen_artifacts()
    tokens = re.findall(
        artifact.token_pattern,
        unicodedata.normalize("NFKC", text).casefold(),
    )
    if not tokens:
        raise ValueError("Pesan tidak menghasilkan token; masukkan teks SMS yang valid.")
    effective = tokens[: artifact.sequence_length]
    ids = np.asarray(
        [artifact.word_to_id.get(token, 1) for token in effective], dtype=np.int32
    )
    encoded = np.zeros((1, artifact.sequence_length), dtype=np.int32)
    encoded[0, : len(ids)] = ids
    return TextEncoding(
        sequence=encoded,
        tokens=tuple(effective),
        unknown_count=int(np.count_nonzero(ids == 1)),
        truncated=len(tokens) > artifact.sequence_length,
    )


@lru_cache(maxsize=1)
def _tensorflow():
    # TensorFlow is intentionally imported only when a prediction is requested.
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    return tf


@lru_cache(maxsize=1)
def _mlp_assets():
    frozen_artifacts()
    import joblib

    tf = _tensorflow()
    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    if len(vectorizer.vocabulary_) != 2000:
        raise RuntimeError("TF-IDF harus memiliki tepat 2000 fitur.")
    model = tf.keras.models.load_model(MODELS / "mlp_baseline.keras", compile=False)
    if model.count_params() != 130177 or tuple(model.input_shape[1:]) != (2000,):
        raise RuntimeError("Arsitektur MLP tidak sama dengan baseline UTS.")
    return vectorizer, model


@lru_cache(maxsize=1)
def _gru_model():
    frozen_artifacts()
    tf = _tensorflow()
    model = tf.keras.models.load_model(MODELS / "gru_attention_tuned.keras", compile=False)
    if model.count_params() != 246433 or tuple(model.input_shape[1:]) != (60,):
        raise RuntimeError("Arsitektur GRU T3 berbeda dari hasil tuning terpilih.")
    return model


@lru_cache(maxsize=1)
def _attention_model():
    frozen_artifacts()
    tf = _tensorflow()
    model = tf.keras.models.load_model(
        MODELS / "gru_attention_tuned_reader.keras", compile=False
    )
    if tuple(model.input_shape[1:]) != (60,):
        raise RuntimeError("Input attention reader tidak berukuran 60 token.")
    return model


def _valid_probability(value: float) -> float:
    if not np.isfinite(value) or not (0 <= value <= 1):
        raise RuntimeError("Model mengembalikan probabilitas yang tidak valid.")
    return value


def predict_mlp(message: str) -> float:
    text = validate_message(message)
    vectorizer, model = _mlp_assets()
    vector = vectorizer.transform([text]).astype(np.float32).toarray()
    if vector.shape != (1, 2000):
        raise RuntimeError("Bentuk input MLP tidak sesuai: seharusnya (1, 2000).")
    probability = float(model.predict(vector, verbose=0).reshape(-1)[0])
    return _valid_probability(probability)


def predict_gru(message: str, *, with_attention: bool = False) -> dict:
    encoding = encode_sms(message)
    probability = float(_gru_model().predict(encoding.sequence, verbose=0).reshape(-1)[0])
    result = {
        "probability_spam": _valid_probability(probability),
        "token_count": len(encoding.tokens),
        "unknown_count": encoding.unknown_count,
        "unknown_fraction": encoding.unknown_fraction,
        "truncated": encoding.truncated,
        "top_attention": [],
    }
    if with_attention:
        weights = np.asarray(_attention_model().predict(encoding.sequence, verbose=0))
        if weights.shape != (1, 1, 60):
            raise RuntimeError(f"Bentuk bobot attention tidak sesuai: {weights.shape}")
        scores = weights[0, 0]
        token_count = len(encoding.tokens)
        if (not np.isfinite(scores).all()
                or not np.isclose(scores[:token_count].sum(), 1.0, atol=1e-3)
                or np.any(np.abs(scores[token_count:]) > 1e-4)):
            raise RuntimeError("Pemeriksaan attention mask / normalisasi gagal.")
        top_positions = sorted(range(token_count), key=lambda idx: -float(scores[idx]))[:5]
        result["top_attention"] = [
            {"token": encoding.tokens[idx], "weight": float(scores[idx]), "position": idx}
            for idx in top_positions
        ]
    return result


def label_and_confidence(probability_spam: float) -> tuple[str, float]:
    label = "Spam" if probability_spam >= THRESHOLD else "Ham"
    confidence = probability_spam if label == "Spam" else (1.0 - probability_spam)
    return label, confidence

