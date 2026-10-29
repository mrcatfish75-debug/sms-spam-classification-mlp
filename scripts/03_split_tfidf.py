r"""Tahap 2A: split data tanpa kebocoran lalu fit TF-IDF hanya pada TRAIN.
Jalankan dari root proyek: .\.venv\Scripts\python.exe scripts\03_split_tfidf.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'sms_en_clean.csv'
SPLITS = ROOT / 'data' / 'splits'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'outputs' / 'reports'
FIGURES = ROOT / 'outputs' / 'figures'
SEED = 42
MIN_SAMPLES = 5000


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(text: str) -> str:
    """Kunci untuk mencari duplikat case/whitespace; teks asli tetap dipertahankan."""
    return re.sub(r'\s+', ' ', text).strip().casefold()


def distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df['label'].value_counts().to_dict()
    return {cls: int(counts.get(cls, 0)) for cls in ('ham', 'spam')}


def main() -> None:
    print('=' * 68)
    print('TAHAP 2A — DATA SPLIT + TRAIN-ONLY TF-IDF')
    print('=' * 68)
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Dataset bersih tidak ditemukan: {SOURCE}')
    df = pd.read_csv(SOURCE, encoding='utf-8-sig', keep_default_na=False)
    needed = {'source_row', 'label', 'text'}
    if not needed.issubset(df.columns):
        raise ValueError(f'Kolom wajib {sorted(needed)}; ditemukan: {df.columns.tolist()}')
    df = df.copy()
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['text'] = df['text'].astype(str)
    bad = (~df['label'].isin(['ham', 'spam'])) | df['text'].str.strip().eq('')
    if bad.any():
        raise ValueError(f'Ada {int(bad.sum())} teks kosong / label invalid. Periksa data sebelum split.')
    if df['source_row'].duplicated().any():
        raise ValueError('source_row tidak unik; cek dataset karena penelusuran sumber akan ambigu.')

    raw_count = len(df)
    df['_key'] = df['text'].map(canonical)
    label_conflicts = df.groupby('_key')['label'].nunique()
    conflict_keys = label_conflicts[label_conflicts > 1].index
    for folder in (SPLITS, MODELS, REPORTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    if len(conflict_keys):
        conflicts = df[df['_key'].isin(conflict_keys)].drop(columns='_key')
        output = REPORTS / 'konflik_normalisasi_perlu_review.csv'
        conflicts.to_csv(output, index=False, encoding='utf-8-sig')
        raise RuntimeError(f'Terdapat {len(conflict_keys)} pesan ternormalisasi dengan label bertentangan. '
                           f'Periksa {output} sebelum lanjut.')
    df = df.drop_duplicates('_key', keep='first').drop(columns='_key').reset_index(drop=True)
    secondary_duplicates = raw_count - len(df)
    if len(df) < MIN_SAMPLES:
        raise RuntimeError(f'Data unik setelah pemeriksaan tambahan tinggal {len(df)} (<{MIN_SAMPLES}). '
                           'Jangan manipulasi data hanya untuk memenuhi minimum.')
    if df['label'].value_counts().min() < 10:
        raise ValueError('Kelas minoritas terlalu kecil untuk stratified split.')

    # Tepat 10% (pembulatan sklearn), lalu 10% dari total sebagai validation.
    trainval, test = train_test_split(df, test_size=0.10, stratify=df['label'], random_state=SEED)
    train, val = train_test_split(trainval, test_size=len(test), stratify=trainval['label'], random_state=SEED)
    train = train.sort_values('source_row').reset_index(drop=True)
    val = val.sort_values('source_row').reset_index(drop=True)
    test = test.sort_values('source_row').reset_index(drop=True)

    # Uji anti-leakage berdasarkan teks kanonis dan ID sumber.
    keys = [set(part['text'].map(canonical)) for part in (train, val, test)]
    rowids = [set(part['source_row']) for part in (train, val, test)]
    assert all(not (keys[i] & keys[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(not (rowids[i] & rowids[j]) for i, j in ((0, 1), (0, 2), (1, 2)))
    assert sum(len(x) for x in (train, val, test)) == len(df)

    # Ini fit hanya pada TRAIN; jangan fit ulang pada validation atau test.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents='unicode',
        token_pattern=r'(?u)\b\w+\b',  # satu karakter seperti 'u' pada SMS tetap dihitung
        ngram_range=(1, 2),
        min_df=2,
        max_features=2000,
        sublinear_tf=True,
        dtype=np.float32
    )
    vectorizer.fit(train['text'])
    x_train = vectorizer.transform(train['text'])
    x_val = vectorizer.transform(val['text'])
    # Tidak melakukan transform(test) di sini: test set dikunci sampai evaluasi akhir.

    paths = {'train': SPLITS / 'sms_train.csv',
             'validation': SPLITS / 'sms_validation.csv',
             'test': SPLITS / 'sms_test.csv'}
    for name, split in (('train', train), ('validation', val), ('test', test)):
        split[['source_row', 'label', 'text']].to_csv(paths[name], index=False, encoding='utf-8-sig')
    vectorizer_path = MODELS / 'tfidf_vectorizer.joblib'
    joblib.dump(vectorizer, vectorizer_path)

    # Grafik rasio kelas tiap partisi
    chart_df = pd.DataFrame({
        name: distribution(split) for name, split in
        (('Train', train), ('Validation', val), ('Test', test))
    }).T[['ham', 'spam']]
    ax = chart_df.plot(kind='bar', figsize=(7, 4), rot=0)
    ax.set_title('Distribusi kelas per partisi')
    ax.set_ylabel('Jumlah SMS')
    ax.legend(title='Label')
    plt.tight_layout()
    plt.savefig(FIGURES / 'distribusi_split.png', dpi=150)
    plt.close()

    manifest = {
        'status': 'SPLIT_TFIDF_PASS',
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': digest_file(SOURCE),
        'input_rows': raw_count,
        'additional_case_whitespace_duplicates_removed': secondary_duplicates,
        'total_final': len(df),
        'seed': SEED,
        'split_policy': 'stratified 80:10:10; fixed seed; dedup before split',
        'train': {'n': len(train), 'class_count': distribution(train), 'path': str(paths['train'].relative_to(ROOT))},
        'validation': {'n': len(val), 'class_count': distribution(val), 'path': str(paths['validation'].relative_to(ROOT))},
        'test_locked': {'n': len(test), 'class_count': distribution(test), 'path': str(paths['test'].relative_to(ROOT))},
        'tfidf': {'path': str(vectorizer_path.relative_to(ROOT)), 'fit_only': 'train',
                  'n_features': len(vectorizer.vocabulary_), 'ngram_range': [1, 2],
                  'max_features': 2000, 'min_df': 2,
                  'train_shape': list(x_train.shape), 'validation_shape': list(x_val.shape)},
        'leakage_checks': 'PASS: text-canonical keys and source_row do not overlap',
        'note': 'TEST belum digunakan untuk tuning atau evaluasi UTS; gunakan VALIDATION.'
    }
    manifest_path = REPORTS / 'split_tfidf_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Data masuk : {raw_count:,}')
    print(f'Duplikat normalisasi tambahan: {secondary_duplicates:,}')
    print(f'Data final : {len(df):,}')
    print(f'Train {len(train):,} {distribution(train)} | Validasi {len(val):,} {distribution(val)} | Test terkunci {len(test):,} {distribution(test)}')
    print(f'Fitur TF-IDF: {len(vectorizer.vocabulary_):,}')
    print('Anti-leakage: PASS')
    print('Manifest:', manifest_path.relative_to(ROOT))
    print('TAHAP 2A PASS — silakan lanjutkan ke 04_train_mlp.py')


if __name__ == '__main__':
    main()
