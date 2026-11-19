"""UAS Tahap 1: train-only tokenization and Word2Vec for SMS sequence models.

Run from the project root:
    .\\.venv\\Scripts\\python.exe scripts\\06_prepare_sequences.py

Reads only TRAIN and VALIDATION partitions. The held-out TEST file is not read.
Requires gensim 4.4.0, pandas, and numpy. Does not require a GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "outputs" / "reports"
SPLIT_MANIFEST = REPORTS / "split_tfidf_manifest.json"

SEED = 42
MAX_VOCAB = 5000  # Includes PAD (0) and UNK (1).
SEQUENCE_LENGTH = 60
EMBEDDING_DIM = 64
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 2
WORD2VEC_EPOCHS = 15
WORD2VEC_WORKERS = 1
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_word_hash(value: str) -> int:
    """Stable initializer for Word2Vec (independent of PYTHONHASHSEED)."""
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little"
    )


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def tokenize(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(TOKEN_PATTERN, normalized)


def load_partition(name: str, expected: dict) -> pd.DataFrame:
    path = SPLITS / f"sms_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split {name} tidak ditemukan: {path}")

    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    mandatory = {"source_row", "label", "text"}
    if not mandatory.issubset(frame.columns):
        raise ValueError(f"Split {name}: kolom wajib {sorted(mandatory)} tidak lengkap.")
    if len(frame) != int(expected["n"]):
        raise ValueError(f"Split {name}: jumlah berubah ({len(frame)} != {expected['n']}).")

    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["text"] = frame["text"].astype(str)
    if (~frame["label"].isin(["spam", "ham"])).any():
        raise ValueError(f"Split {name}: ada label selain spam/ham.")
    if frame["text"].str.strip().eq("").any():
        raise ValueError(f"Split {name}: ada pesan kosong.")
    if frame["source_row"].duplicated().any():
        raise ValueError(f"Split {name}: source_row tidak unik.")
    actual_counts = frame["label"].value_counts().to_dict()
    expected_counts = expected["class_count"]
    for label in ("ham", "spam"):
        if int(actual_counts.get(label, 0)) != int(expected_counts[label]):
            raise ValueError(f"Split {name}: distribusi {label} berubah.")
    return frame


def encode(tokens: list[str], word_to_id: dict[str, int]) -> np.ndarray:
    ids = [word_to_id.get(token, 1) for token in tokens[:SEQUENCE_LENGTH]]
    out = np.zeros(SEQUENCE_LENGTH, dtype=np.int32)
    out[: len(ids)] = ids
    return out


def coverage(token_lists: list[list[str]], word_to_id: dict[str, int]) -> dict:
    observed = sum(len(tokens) for tokens in token_lists)
    unknown = sum(sum(token not in word_to_id for token in tokens) for tokens in token_lists)
    return {
        "tokens_total": int(observed),
        "tokens_unknown": int(unknown),
        "oov_ratio": round(unknown / observed, 6) if observed else 0.0,
    }


def main() -> None:
    print("=" * 70)
    print("UAS TAHAP 1 - TOKENISASI, WORD2VEC, DAN SEKUENSIAL TRAIN-ONLY")
    print("=" * 70)

    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError("Manifest UTS tidak ditemukan; periksa output 03_split_tfidf.py.")
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "SPLIT_TFIDF_PASS":
        raise RuntimeError("Manifest split UTS belum berstatus PASS.")

    train = load_partition("train", manifest["train"])
    validation = load_partition("validation", manifest["validation"])

    train_ids = set(train["source_row"])
    val_ids = set(validation["source_row"])
    if train_ids & val_ids:
        raise RuntimeError("LEAKAGE: source_row TRAIN dan VALIDATION beririsan.")
    train_keys = set(train["text"].map(canonical))
    val_keys = set(validation["text"].map(canonical))
    if train_keys & val_keys:
        raise RuntimeError("LEAKAGE: teks kanonis TRAIN dan VALIDATION beririsan.")

    train_tokens = [tokenize(text) for text in train["text"]]
    val_tokens = [tokenize(text) for text in validation["text"]]
    if any(not tokens for tokens in train_tokens + val_tokens):
        raise ValueError("Ada SMS yang kosong setelah tokenisasi; periksa pola token.")

    # The vocabulary and Word2Vec are built EXCLUSIVELY from training SMS.
    counts = Counter(token for tokens in train_tokens for token in tokens)
    selected = sorted(
        ((word, count) for word, count in counts.items() if count >= WORD2VEC_MIN_COUNT),
        key=lambda item: (-item[1], item[0]),
    )[: MAX_VOCAB - 2]
    vocabulary = [PAD_TOKEN, UNK_TOKEN, *(word for word, _ in selected)]
    word_to_id = {word: idx for idx, word in enumerate(vocabulary)}
    if len(vocabulary) < 20:
        raise ValueError("Vocabulary terlalu kecil untuk eksperimen: < 20 token.")

    w2v = Word2Vec(
        sentences=train_tokens,
        vector_size=EMBEDDING_DIM,
        window=WORD2VEC_WINDOW,
        min_count=WORD2VEC_MIN_COUNT,
        workers=WORD2VEC_WORKERS,
        seed=SEED,
        sg=1,  # Skip-gram.
        epochs=WORD2VEC_EPOCHS,
        hashfxn=stable_word_hash,
    )

    rng = np.random.default_rng(SEED)
    embedding = np.zeros((len(vocabulary), EMBEDDING_DIM), dtype=np.float32)
    embedding[1] = rng.normal(0.0, 0.01, EMBEDDING_DIM).astype(np.float32)
    missing_vectors: list[str] = []
    for word, idx in word_to_id.items():
        if idx < 2:
            continue
        if word not in w2v.wv:
            missing_vectors.append(word)
        else:
            embedding[idx] = w2v.wv[word]
    if missing_vectors:
        raise RuntimeError(f"Word2Vec kehilangan {len(missing_vectors)} kata vocabulary.")

    x_train = np.stack([encode(tokens, word_to_id) for tokens in train_tokens])
    x_val = np.stack([encode(tokens, word_to_id) for tokens in val_tokens])
    y_train = (train["label"] == "spam").to_numpy(dtype=np.int8)
    y_val = (validation["label"] == "spam").to_numpy(dtype=np.int8)
    id_train = train["source_row"].to_numpy(dtype=np.int32)
    id_val = validation["source_row"].to_numpy(dtype=np.int32)

    for folder in (PROCESSED, MODELS, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED / "sequences_train.npz", x=x_train, y=y_train, source_row=id_train
    )
    np.savez_compressed(
        PROCESSED / "sequences_validation.npz", x=x_val, y=y_val, source_row=id_val
    )
    np.save(MODELS / "word2vec_embedding_matrix.npy", embedding)
    w2v.wv.save(str(MODELS / "word2vec_train_only.kv"))
    (MODELS / "sequence_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tokenizer_config = {
        "standardization": "NFKC + casefold",
        "token_pattern": TOKEN_PATTERN,
        "max_vocab": MAX_VOCAB,
        "vocabulary_size": len(vocabulary),
        "sequence_length": SEQUENCE_LENGTH,
        "padding": "post; token index 0",
        "truncation": "post",
        "unknown_token_index": 1,
        "embedding_dim": EMBEDDING_DIM,
    }
    (MODELS / "sequence_tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "status": "SEQUENCE_WORD2VEC_PASS",
        "dataset": "UCI SMS Spam Collection - original English",
        "scope": "TRAIN_AND_VALIDATION_ONLY__TEST_NOT_READ",
        "source_manifest_status": manifest["status"],
        "seed": SEED,
        "train_count": len(train),
        "validation_count": len(validation),
        "train_sha256": sha256_file(SPLITS / "sms_train.csv"),
        "validation_sha256": sha256_file(SPLITS / "sms_validation.csv"),
        "train_validation_overlap_count": 0,
        "tokenization": tokenizer_config,
        "word2vec": {
            "type": "Skip-gram trained from scratch on TRAIN only (not external pretraining)",
            "vector_size": EMBEDDING_DIM,
            "window": WORD2VEC_WINDOW,
            "min_count": WORD2VEC_MIN_COUNT,
            "workers": WORD2VEC_WORKERS,
            "epochs": WORD2VEC_EPOCHS,
            "corpus_documents": len(train_tokens),
        },
        "sequence_train_shape": list(x_train.shape),
        "sequence_validation_shape": list(x_val.shape),
        "embedding_shape": list(embedding.shape),
        "token_coverage_train": coverage(train_tokens, word_to_id),
        "token_coverage_validation": coverage(val_tokens, word_to_id),
        "train_messages_truncated": int(sum(len(x) > SEQUENCE_LENGTH for x in train_tokens)),
        "validation_messages_truncated": int(sum(len(x) > SEQUENCE_LENGTH for x in val_tokens)),
        "padding_vector_is_zero": bool(np.all(embedding[0] == 0)),
        "test_status": "LOCKED_NOT_READ",
        "privacy_note": "Report and model files exclude SMS raw text. Do not commit private CSV/NPZ.",
    }
    report_path = REPORTS / "uas_sequence_word2vec_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"TRAIN: {len(train):,} | VALIDATION: {len(validation):,}")
    print(f"Vocabulary: {len(vocabulary):,} (PAD=0, UNK=1)")
    print(f"Sequence shapes: {x_train.shape} | {x_val.shape}")
    print(f"Embedding matrix: {embedding.shape}")
    print(f"Validation OOV ratio: {report['token_coverage_validation']['oov_ratio']:.2%}")
    print(f"Validation SMS truncated (> {SEQUENCE_LENGTH} tokens): {report['validation_messages_truncated']}")
    print(f"Manifest: {report_path.relative_to(ROOT)}")
    print("TRAIN/VALIDATION ANTI-LEAKAGE: PASS")
    print("TEST: NOT READ - tetap terkunci sampai evaluasi final UAS.")
    print("UAS TAHAP 1 PASS - siap membangun GRU.")


if __name__ == "__main__":
    main()
