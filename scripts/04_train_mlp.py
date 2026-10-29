r"""Tahap 2B: melatih MLP baseline TF-IDF, evaluasi di validation ONLY.
Jalankan dari root proyek: .\.venv\Scripts\python.exe scripts\04_train_mlp.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'data' / 'splits'
MODELS = ROOT / 'models'
REPORTS = ROOT / 'outputs' / 'reports'
FIGURES = ROOT / 'outputs' / 'figures'
SEED = 42


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 6),
        'precision_spam': round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        'recall_spam': round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        'f1_spam': round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def main() -> None:
    print('=' * 68)
    print('TAHAP 2B — TRAINING MLP BASELINE + EVALUASI VALIDATION')
    print('=' * 68)
    for folder in (MODELS, REPORTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS / 'split_tfidf_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Jalankan 03_split_tfidf.py terlebih dahulu.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'SPLIT_TFIDF_PASS':
        raise RuntimeError('Manifest split tidak PASS.')
    train = pd.read_csv(SPLITS / 'sms_train.csv', keep_default_na=False)
    val = pd.read_csv(SPLITS / 'sms_validation.csv', keep_default_na=False)
    assert len(train) == manifest['train']['n']
    assert len(val) == manifest['validation']['n']
    assert set(train['label'].unique()) <= {'ham', 'spam'}
    assert set(val['label'].unique()) <= {'ham', 'spam'}
    # Jangan memuat atau mengevaluasi sms_test.csv.
    y_train = (train['label'].to_numpy() == 'spam').astype(np.float32)
    y_val = (val['label'].to_numpy() == 'spam').astype(np.int32)
    vectorizer = joblib.load(MODELS / 'tfidf_vectorizer.joblib')  # hanya file dari script kita sendiri
    x_train = vectorizer.transform(train['text']).astype(np.float32).toarray()
    x_val = vectorizer.transform(val['text']).astype(np.float32).toarray()
    print('Matriks TF-IDF:', x_train.shape, x_val.shape)
    print('Penyimpanan matriks TRAIN ~%.1f MB' % (x_train.nbytes / (1024 * 1024)))

    # Import framework setelah data siap supaya tahap split tetap ringan.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError('TensorFlow belum terinstal. Jalankan pip install -r requirements-tahap2.txt') from exc
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # Sudah diinisialisasi oleh runtime TensorFlow.
    tf.keras.backend.clear_session()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1],), name='tfidf_features'),
        tf.keras.layers.Dense(64, activation='relu', name='hidden_64'),
        tf.keras.layers.Dropout(0.20, name='dropout_020'),
        tf.keras.layers.Dense(32, activation='relu', name='hidden_32'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='spam_probability')
    ], name='sms_spam_mlp_uts')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)]
    hist = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=15, batch_size=32, callbacks=callbacks, shuffle=True, verbose=2)

    proba = model.predict(x_val, batch_size=64, verbose=0).reshape(-1)
    y_pred = (proba >= 0.50).astype(np.int32)
    mlp_metrics = get_metrics(y_val, y_pred)
    mlp_metrics['pr_auc_average_precision'] = round(float(average_precision_score(y_val, proba)), 6)
    # Referensi naif: semua SMS diprediksi ham, bukan Deep Learning.
    majority = np.zeros_like(y_val)
    naive_metrics = get_metrics(y_val, majority)
    report_text = classification_report(
        y_val, y_pred, labels=[0, 1], target_names=['ham', 'spam'], zero_division=0)
    report_path = REPORTS / 'classification_report_validation.txt'
    report_path.write_text(report_text, encoding='utf-8')

    model_path = MODELS / 'mlp_baseline.keras'
    model.save(model_path)
    history_df = pd.DataFrame(hist.history)
    history_df.insert(0, 'epoch', np.arange(1, len(history_df) + 1))
    history_df.to_csv(REPORTS / 'mlp_training_history.csv', index=False)
    results = {
        'status': 'MLP_TRAINED_VALIDATION_ONLY',
        'dataset_rows': manifest['total_final'],
        'train_rows': len(train),
        'validation_rows': len(val),
        'test_rows_locked': manifest['test_locked']['n'],
        'features': x_train.shape[1],
        'model': 'TF-IDF -> Dense64 ReLU -> Dropout0.2 -> Dense32 ReLU -> Sigmoid',
        'optimizer': 'Adam(learning_rate=0.001)',
        'loss': 'binary_crossentropy',
        'batch_size': 32, 'max_epochs': 15,
        'epochs_run': int(len(history_df)),
        'threshold': 0.5,
        'validation_mlp': mlp_metrics,
        'validation_majority_ham_reference': naive_metrics,
        'confusion_matrix_validation_labels_0ham_1spam': confusion_matrix(
            y_val, y_pred, labels=[0, 1]).tolist(),
        'test_status': 'LOCKED_NOT_EVALUATED',
        'caveats': ['Data memiliki class imbalance',
                    'Sumber SMS lama berbahasa Inggris; tidak membuktikan generalisasi ke SMS Indonesia/modern',
                    'Hasil hanya validation; belum final TEST/UAS']
    }
    results_path = REPORTS / 'hasil_validasi_mlp.json'
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    # Grafik loss dan akurasi untuk laporan.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_df['epoch'], history_df['loss'], label='Train')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Binary crossentropy', title='Loss')
    axes[0].legend()
    axes[1].plot(history_df['epoch'], history_df['accuracy'], label='Train')
    axes[1].plot(history_df['epoch'], history_df['val_accuracy'], label='Validation')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / 'kurva_training_mlp.png', dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_val, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0, 1], ['Ham', 'Spam'])
    ax.set_yticks([0, 1], ['Ham', 'Spam'])
    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Aktual')
    ax.set_title('Confusion matrix — VALIDATION')
    for i in (0, 1):
        for j in (0, 1):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(FIGURES / 'confusion_matrix_validation.png', dpi=150)
    plt.close(fig)

    print('\nHASIL MLP (VALIDATION — BUKAN TEST)')
    for key, value in mlp_metrics.items():
        print(f'{key}: {value}')
    print('Majority-ham accuracy reference:', naive_metrics['accuracy'])
    print('\n' + report_text)
    print('Model tersimpan:', model_path.relative_to(ROOT))
    print('Hasil evaluasi:', results_path.relative_to(ROOT))
    print('TEST TIDAK DIBACA / TIDAK DIGUNAKAN')


if __name__ == '__main__':
    main()
