r"""Gradio 6 UI R02 for the frozen SMS Spam UTS/UAS inference models.

Windows, from project root:
    .\.venv\Scripts\python.exe app.py

The UI layer calls only the unchanged src/uas_inference.py interface. It does
not read dataset splits, retrain models, change thresholds, or persist SMS.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from src.uas_inference import (
    THRESHOLD,
    frozen_artifacts,
    predict_gru,
    predict_mlp,
    validate_message,
)
from ui.r03_helpers import guard_html, invalidated_html


MODEL_GRU = "GRU + Attention (UAS)"
MODEL_MLP = "MLP baseline (UTS)"
MODEL_BOTH = "Bandingkan dua model"

SAMPLE_NORMAL = "Hi, can we meet for coffee tomorrow afternoon?"
SAMPLE_PROMO = "Congratulations! Claim your free prize by calling our number today!"

CSS_PATH = Path(__file__).resolve().parent / "ui" / "uas_theme.css"

EMPTY_RESULT = (
    '<section class="result-banner result-waiting" aria-live="polite">'
    '<span class="result-eyebrow">Hasil analisis</span>'
    '<h2>Belum ada pesan yang dianalisis</h2>'
    '<p>Masukkan SMS berbahasa Inggris, pilih model, lalu klik '
    '<strong>Analisis pesan</strong>.</p>'
    '</section>'
)


def percent(probability: float) -> str:
    """Indonesian-facing formatting only; does not round the decision value."""
    return f"{probability * 100:.2f}".replace(".", ",") + "%"


def friendly_label(probability: float) -> tuple[str, str, str]:
    if probability >= THRESHOLD:
        return (
            "Terindikasi spam",
            "Pesan memiliki pola yang dikenali model sebagai spam.",
            "is-spam",
        )
    return (
        "Pesan normal (Ham)",
        "Pesan tidak terindikasi spam menurut model ini.",
        "is-ham",
    )


def score_card(model_name: str, probability: float) -> str:
    """Accessible, identical score treatment for a single or compared model."""
    label, _, status_class = friendly_label(probability)
    label = html.escape(label)
    name = html.escape(model_name)
    marker_position = probability * 100  # Only used to render an exact-width bar.
    return (
        f'<section class="score-card {status_class}" '
        f'aria-label="{name}: {label}, probabilitas spam {percent(probability)}">'
        f'<div class="score-model">{name}</div>'
        f'<div class="score-state"><span class="status-dot" aria-hidden="true"></span>{label}</div>'
        '<div class="score-main">'
        '<span class="score-caption">Perkiraan probabilitas spam</span>'
        f'<strong class="score-value">{percent(probability)}</strong>'
        '</div>'
        '<div class="score-track" role="img" '
        f'aria-label="Skala 0 sampai 100 persen. Nilai spam {percent(probability)}, '
        'ambang keputusan 50 persen">'
        f'<div class="score-fill" style="width:{marker_position:.6f}%"></div>'
        '<div class="score-threshold" aria-hidden="true"></div>'
        '</div>'
        '<div class="score-scale"><span>0%</span><span>Ambang 50%</span><span>100%</span></div>'
        '</section>'
    )


def attention_view(result: dict, requested: bool) -> str:
    """Rank actual attention weights, with relative-only bars for readability."""
    if not requested:
        return ""
    top = result.get("top_attention", [])
    if not top:
        return '<p class="technical-note">Bobot Attention tidak tersedia.</p>'

    largest = max(float(item["weight"]) for item in top)
    rows = []
    for rank, item in enumerate(top, start=1):
        word = html.escape(str(item["token"]), quote=True)
        weight = float(item["weight"])
        relative = 100 * weight / largest if largest > 0 else 0
        rows.append(
            '<div class="attention-row">'
            f'<span class="attention-rank">{rank:02d}</span>'
            f'<code class="attention-token">{word}</code>'
            '<div class="attention-bar" aria-hidden="true">'
            f'<div class="attention-fill" style="width:{relative:.3f}%"></div></div>'
            f'<span class="attention-weight">{weight:.3f}</span>'
            '</div>'
        )

    return (
        '<section class="attention-list">'
        '<h3>5 token dengan bobot Attention tertinggi</h3>'
        '<p class="technical-note">Panjang batang dibandingkan dengan token berbobot '
        'tertinggi; angka di kanan adalah bobot asli dari model.</p>'
        + "".join(rows)
        + '<p class="attention-disclaimer">Bobot Attention menunjukkan perhatian relatif '
        'model pada token, bukan bukti bahwa token tersebut menyebabkan prediksi.</p>'
        '</section>'
    )


def result_banner(displays: list[tuple[str, float]]) -> str:
    if len(displays) == 1:
        model, probability = displays[0]
        label, description, state_class = friendly_label(probability)
        return (
            f'<section class="result-banner {state_class}" aria-live="polite">'
            '<span class="result-eyebrow">Hasil analisis</span>'
            f'<h2><span class="status-dot" aria-hidden="true"></span>{html.escape(label)}</h2>'
            f'<p>{html.escape(description)}</p>'
            f'<span class="result-model">Model: {html.escape(model)}</span>'
            '</section>'
        )

    agree = (displays[0][1] >= THRESHOLD) == (displays[1][1] >= THRESHOLD)
    title = "Kedua model memberikan label yang sama" if agree else "Kedua model berbeda pendapat"
    description = (
        "Bandingkan kedua hasil di bawah. Kesamaan prediksi bukan jaminan keamanan."
        if agree else
        "Periksa keluaran masing-masing model. Hasil yang berbeda bukan ukuran siapa yang selalu lebih baik."
    )
    state_class = "is-agree" if agree else "is-disagree"
    return (
        f'<section class="result-banner {state_class}" aria-live="polite">'
        '<span class="result-eyebrow">Perbandingan dua model</span>'
        f'<h2>{html.escape(title)}</h2>'
        f'<p>{html.escape(description)}</p>'
        '</section>'
    )


def diagnostics(gru_result: dict | None, compared: bool) -> str:
    parts = [
        '<section class="technical-copy">'
        '<h3>Cara membaca hasil</h3>'
        '<p>Model memperkirakan probabilitas SMS sebagai spam. Nilai '
        '<strong>50% atau lebih</strong> diberi label Spam; di bawah 50% diberi label Ham. '
        'Angka tersebut bukan jaminan bahwa pesan aman dan belum diuji sebagai probabilitas terkalibrasi.</p>'
    ]
    if compared:
        parts.append(
            '<p>Perbandingan menunjukkan keluaran pada SMS yang sama, bukan '
            'perbandingan akurasi model pada dataset evaluasi.</p>'
        )
    if gru_result is not None:
        parts.append(
            '<h3>Rincian pemrosesan teks GRU</h3>'
            f'<p>Token dipakai: <strong>{gru_result["token_count"]}</strong> · '
            f'Token di luar kosakata: <strong>{gru_result["unknown_count"]}</strong> '
            f'({percent(gru_result["unknown_fraction"])})</p>'
        )
        if gru_result["truncated"]:
            parts.append('<p>Pesan dipotong setelah 60 token sesuai konfigurasi pelatihan.</p>')
        if gru_result["unknown_fraction"] > 0.30:
            parts.append('<p>Banyak token di luar kosakata; hasil perlu ditafsirkan dengan lebih hati-hati.</p>')
    parts.append('</section>')
    return "".join(parts)


def analyze(message: str, selected: str, show_attention: bool):
    """Four UI outputs; read-only frozen inference; no input is logged or saved."""
    try:
        validate_message(message)
    except ValueError as error:
        error_text = html.escape(str(error), quote=True)
        return (
            '<section class="result-banner result-invalid" role="alert">'
            '<span class="result-eyebrow">Periksa masukan</span>'
            f'<h2>Pesan belum dapat dianalisis</h2><p>{error_text}</p></section>',
            "", "", "",
        )

    if selected not in {MODEL_GRU, MODEL_MLP, MODEL_BOTH}:
        raise ValueError("Pilihan model tidak dikenal.")

    need_gru = selected in {MODEL_GRU, MODEL_BOTH}
    need_mlp = selected in {MODEL_MLP, MODEL_BOTH}
    gru_result = predict_gru(message, with_attention=show_attention) if need_gru else None
    mlp_probability = predict_mlp(message) if need_mlp else None

    displayed = []
    if gru_result is not None:
        displayed.append((MODEL_GRU, gru_result["probability_spam"]))
    if mlp_probability is not None:
        displayed.append((MODEL_MLP, mlp_probability))

    summary_html = result_banner(displayed)
    score_html = (
        '<div class="score-grid two-scores">' if len(displayed) == 2 else '<div class="score-grid">'
    ) + "".join(score_card(name, value) for name, value in displayed) + '</div>'
    explanation_html = diagnostics(gru_result, compared=len(displayed) == 2)

    if selected == MODEL_MLP and show_attention:
        attention_html = (
            '<p class="technical-note">Visualisasi Attention hanya tersedia '
            'pada GRU + Attention (UAS).</p>'
        )
    else:
        attention_html = attention_view(gru_result or {}, show_attention)
    return summary_html, score_html, explanation_html, attention_html


def build_app():
    # Fail closed before presenting any UI if frozen artifacts no longer match.
    frozen_artifacts()
    if not CSS_PATH.is_file():
        raise FileNotFoundError("File tampilan UI R02 tidak ditemukan: ui/uas_theme.css")

    with gr.Blocks(title="Klasifikasi SMS Spam · UAS", analytics_enabled=False) as demo:
        gr.HTML(
            '<header class="sms-hero">'
            '<div class="sms-overline">Proyek Deep Learning <span aria-hidden="true">/</span> Jalur B</div>'
            '<h1>Klasifikasi SMS Spam</h1>'
            '<p>Periksa pesan berbahasa Inggris menggunakan dua model penelitian: '
            'MLP baseline (UTS) dan GRU + Attention (UAS). '
            'Hasil adalah perkiraan model, bukan pemeriksaan keamanan.</p>'
            '</header>'
        )

        with gr.Row(elem_id="sms-workbench", equal_height=False):
            with gr.Column(scale=5, min_width=320, elem_id="sms-input-panel"):
                gr.HTML(
                    '<div class="panel-head"><span class="panel-step">01</span>'
                    '<div><h2>Masukkan pesan</h2><p>Satu SMS berbahasa Inggris</p></div></div>'
                )
                message = gr.Textbox(
                    label="Teks SMS", lines=6, max_lines=12,
                    placeholder="Ketik atau tempel teks SMS di sini...",
                    max_length=2000,
                    elem_id="sms-message",
                )
                gr.HTML('<p class="field-heading">Contoh pesan</p>')
                with gr.Row(elem_id="sms-examples"):
                    example_normal = gr.Button("Pesan biasa", size="sm", variant="secondary")
                    example_spam = gr.Button("Pesan promosi", size="sm", variant="secondary")

                selected = gr.Radio(
                    choices=[MODEL_GRU, MODEL_MLP, MODEL_BOTH],
                    value=MODEL_GRU,
                    label="Pilih model analisis",
                    elem_id="sms-model-select",
                )
                with gr.Accordion("Opsi interpretasi", open=False, elem_id="sms-attention-option"):
                    show_attention = gr.Checkbox(
                        label="Tampilkan 5 token dengan bobot Attention tertinggi",
                        value=False,
                    )
                    gr.HTML(
                        '<p class="technical-note">Tersedia untuk GRU. Bobot token '
                        'digunakan sebagai informasi pendukung, bukan bukti sebab-akibat.</p>'
                    )

                button = gr.Button(
                    "Analisis pesan", variant="primary", elem_id="sms-primary-action"
                )

            with gr.Column(scale=6, min_width=360, elem_id="sms-result-panel"):
                gr.HTML(
                    '<div class="panel-head"><span class="panel-step">02</span>'
                    '<div><h2>Baca hasil</h2><p>Label, skor, dan interpretasi</p></div></div>'
                )
                summary = gr.HTML(value=EMPTY_RESULT, elem_id="sms-result-summary")
                probability = gr.HTML(elem_id="sms-result-score")
                gr.HTML(
                    '<p class="safety-inline">Prediksi statistik ini tidak menjamin '
                    'bahwa pesan aman. Jangan mengikuti tautan atau mengirim data '
                    'sensitif hanya berdasarkan hasil model.</p>'
                )
                with gr.Accordion("Cara membaca hasil & rincian teknis", open=False,
                                  elem_id="sms-technical-panel"):
                    explanation = gr.HTML(elem_id="sms-explanation")
                    attention = gr.HTML(elem_id="sms-attention")

        example_normal.click(
            fn=lambda: SAMPLE_NORMAL, outputs=message, show_progress="hidden"
        )
        example_spam.click(
            fn=lambda: SAMPLE_PROMO, outputs=message, show_progress="hidden"
        )
        _r03_analysis_event = button.click(
            fn=guard_html(analyze, html_positions=(0, 1, 2, 3)),
            inputs=[message, selected, show_attention],
            outputs=[summary, probability, explanation, attention],
            show_progress="minimal",
        )

        # R03_SAFE_PATCH_APPLIED
        def _r03_clear_results():
            return (gr.update(value=invalidated_html()), gr.update(value=""), gr.update(value=""), gr.update(value=""))
        message.change(fn=_r03_clear_results, outputs=[summary, probability, explanation, attention], queue=False, show_progress="hidden", cancels=[_r03_analysis_event])
        selected.change(fn=_r03_clear_results, outputs=[summary, probability, explanation, attention], queue=False, show_progress="hidden", cancels=[_r03_analysis_event])
        show_attention.change(fn=_r03_clear_results, outputs=[summary, probability, explanation, attention], queue=False, show_progress="hidden", cancels=[_r03_analysis_event])

        gr.HTML(
            '<footer class="sms-footer">'
            '<strong>Batasan penggunaan</strong>'
            '<p>Korpus pelatihan berasal dari SMS Inggris historis. Kinerja pada '
            'SMS Indonesia atau pola penipuan baru tidak terjamin. Jangan memasukkan '
            'nomor pribadi, OTP, atau informasi rahasia. Kode proyek tidak menyimpan '
            'teks masukan, tetapi penyedia hosting tetap dapat memproses permintaan.</p>'
            '</footer>'
        )

    demo.queue(default_concurrency_limit=1, max_size=8)
    return demo


if __name__ == "__main__":
    server_name = os.environ.get(
        "GRADIO_SERVER_NAME", "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    )
    app = build_app()
    app.launch(
        server_name=server_name,
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        inbrowser=(server_name == "127.0.0.1"),
        show_error=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css_paths=[str(CSS_PATH)],
    )
