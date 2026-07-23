"""
Generate light-background architecture diagrams for DATN thesis.
Labels match current code: English caption + MiniLM freeze + projection
(not the old 6-ID embedding scheme).

Run from anywhere:
  python DATN_Form_Xu__ng__1_/figures/gen_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent

# Thesis figure style (print-friendly white background)
BG = "#ffffff"
BOX_FACE = "#f4f7fb"
BOX_EDGE = "#2c3e50"
BOX_FACE_ACCENT = "#e8f0fe"  # condition / text path
BOX_FACE_LOSS = "#fdecea"
BOX_FACE_OK = "#e8f5e9"
BOX_FACE_WARN = "#fff8e6"
TEXT = "#1a1a1a"
ARROW = "#34495e"
TITLE = "#1a252f"
FS = 9
FS_TITLE = 11
FS_SMALL = 8


def _setup(fig_w, fig_h):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, face=BOX_FACE, fs=FS, lw=1.2, radius=0.02):
    """x,y = center; w,h in data coords."""
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0.3,rounding_size={radius * min(w, h)}",
        facecolor=face,
        edgecolor=BOX_EDGE,
        linewidth=lw,
        mutation_aspect=1,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color=TEXT,
        fontfamily="DejaVu Sans",
        zorder=3,
        linespacing=1.25,
        wrap=False,
    )
    return (x, y, w, h)


def arrow(ax, p1, p2, rad=0.0, lw=1.3):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=ARROW,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(a)


def edge_right(b):
    x, y, w, h = b
    return (x + w / 2, y)


def edge_left(b):
    x, y, w, h = b
    return (x - w / 2, y)


def edge_top(b):
    x, y, w, h = b
    return (x, y + h / 2)


def edge_bot(b):
    x, y, w, h = b
    return (x, y - h / 2)


def footnote(ax, text, y=4):
    ax.text(
        50,
        y,
        text,
        ha="center",
        va="center",
        fontsize=FS_SMALL,
        color="#555",
        style="italic",
    )


def save(fig, name: str):
    path = OUT / name
    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor=BG,
        edgecolor="none",
        pad_inches=0.25,
    )
    plt.close(fig)
    print(f"  wrote {path.name}")


# ---------------------------------------------------------------------------
# 1. Music Transformer train
# ---------------------------------------------------------------------------
def gen_transformer():
    fig, ax = _setup(13.0, 5.4)
    ax.set_title(
        "Music Transformer — training pipeline (English text + MiniLM)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b_midi = box(ax, 9, 75, 13, 13, "MIDI file")
    b_tok = box(ax, 28, 75, 18, 15, "Tokenizer REMI\nNOTE_ON / TIME_SHIFT …")
    b_emb = box(ax, 50, 75, 16, 13, "Token Embedding\n+ RoPE")
    b_dec = box(ax, 72, 75, 18, 16, "×6 DecoderBlock\ncross-attn + FFN")
    b_log = box(ax, 92, 75, 12, 13, "Logits\nvocab")
    b_loss = box(ax, 92, 45, 13, 13, "Loss CE\nnext token", face=BOX_FACE_LOSS)

    b_lab = box(ax, 12, 28, 16, 15, "labels.json\n6 attributes", face=BOX_FACE_ACCENT)
    b_cap = box(ax, 34, 28, 18, 15, "English caption\n(template)", face=BOX_FACE_ACCENT)
    b_min = box(
        ax, 56, 28, 20, 16, "MiniLM (freeze)\n+ projection", face=BOX_FACE_ACCENT
    )
    b_cond = box(ax, 78, 28, 14, 14, "cond c\n(B,1,d)", face=BOX_FACE_OK)

    arrow(ax, edge_right(b_midi), edge_left(b_tok))
    arrow(ax, edge_right(b_tok), edge_left(b_emb))
    arrow(ax, edge_right(b_emb), edge_left(b_dec))
    arrow(ax, edge_right(b_dec), edge_left(b_log))
    arrow(ax, edge_bot(b_log), edge_top(b_loss))

    arrow(ax, edge_right(b_lab), edge_left(b_cap))
    arrow(ax, edge_right(b_cap), edge_left(b_min))
    arrow(ax, edge_right(b_min), edge_left(b_cond))
    arrow(ax, edge_top(b_cond), edge_bot(b_dec))

    footnote(ax, "Shared I/O: English sentence → c · Music path: REMI tokens · CE loss")
    save(fig, "transformer.png")


# ---------------------------------------------------------------------------
# 2. Diffusion train
# ---------------------------------------------------------------------------
def gen_train_diffusion():
    fig, ax = _setup(13.0, 5.6)
    ax.set_title(
        "Piano-roll Diffusion — training pipeline (English text + MiniLM)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b_midi = box(ax, 9, 80, 12, 12, "MIDI")
    b_pr = box(ax, 28, 80, 20, 15, "midi_to_pianoroll\n88×256 @ 24 Hz")
    b_x0 = box(ax, 50, 80, 12, 12, "x0 clean")
    b_noise = box(ax, 50, 52, 18, 13, "Add noise → x_t")
    b_t = box(ax, 28, 52, 14, 13, "sample t\n0…T−1")
    b_unet = box(ax, 74, 52, 18, 16, "Conditional UNet\nFiLM + mid-attn")
    b_eps = box(ax, 93, 52, 11, 12, "pred ε")
    b_loss = box(ax, 93, 28, 12, 12, "Loss MSE", face=BOX_FACE_LOSS)

    b_lab = box(ax, 12, 18, 14, 14, "labels /\ncaption", face=BOX_FACE_ACCENT)
    b_min = box(
        ax, 36, 18, 20, 15, "MiniLM (freeze)\n+ projection", face=BOX_FACE_ACCENT
    )
    b_cond = box(ax, 60, 18, 12, 12, "cond c", face=BOX_FACE_OK)

    arrow(ax, edge_right(b_midi), edge_left(b_pr))
    arrow(ax, edge_right(b_pr), edge_left(b_x0))
    arrow(ax, edge_bot(b_x0), edge_top(b_noise))
    arrow(ax, edge_right(b_t), edge_left(b_noise))
    arrow(ax, edge_right(b_noise), edge_left(b_unet))
    ax.annotate(
        "",
        xy=edge_left(b_unet),
        xytext=edge_bot(b_t),
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW,
            lw=1.2,
            connectionstyle="arc3,rad=0.25",
        ),
    )
    ax.text(38, 38, "t", fontsize=FS_SMALL, color=ARROW)
    arrow(ax, edge_right(b_unet), edge_left(b_eps))
    arrow(ax, edge_bot(b_eps), edge_top(b_loss))
    arrow(ax, edge_right(b_lab), edge_left(b_min))
    arrow(ax, edge_right(b_min), edge_left(b_cond))
    arrow(ax, edge_top(b_cond), edge_bot(b_unet))

    footnote(
        ax,
        "CFG train: cond-drop → c = 0 with p≈0.1 · EMA on UNet + projection · cosine T=1000",
    )
    save(fig, "train-diffusion.png")


# ---------------------------------------------------------------------------
# 3. Decoder block
# ---------------------------------------------------------------------------
def gen_decoder_block():
    fig, ax = _setup(6.5, 9.0)
    ax.set_title(
        "One DecoderBlock (Music Transformer)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=6,
        fontweight="bold",
    )

    nodes = []
    steps = [
        ("x: hidden token sequence", BOX_FACE),
        ("RMSNorm", BOX_FACE),
        ("Masked Self-Attention\nGQA + RoPE", BOX_FACE),
        ("x = x + SA", BOX_FACE),
        ("RMSNorm", BOX_FACE),
        ("Cross-Attention\n(K,V from cond c)", BOX_FACE_ACCENT),
        ("x = x + CA", BOX_FACE),
        ("RMSNorm", BOX_FACE),
        ("SwiGLU FFN", BOX_FACE),
        ("x = x + FFN", BOX_FACE),
    ]
    for i, (txt, face) in enumerate(steps):
        yy = 90 - i * 8.5
        b = box(ax, 55, yy, 42, 7, txt, face=face, fs=FS)
        nodes.append(b)
        if i > 0:
            arrow(ax, edge_bot(nodes[i - 1]), edge_top(b))

    b_cond = box(
        ax,
        18,
        90 - 5 * 8.5,
        28,
        12,
        "cond c from\nTextPromptEncoder\n(MiniLM + proj)",
        face=BOX_FACE_ACCENT,
        fs=FS_SMALL,
    )
    arrow(ax, edge_right(b_cond), edge_left(nodes[5]))
    save(fig, "decoder-block.png")


# ---------------------------------------------------------------------------
# 4. Conditioning (cross-attn) — dieu_Kien_hoa
# ---------------------------------------------------------------------------
def gen_conditioning():
    fig, ax = _setup(11, 4.8)
    ax.set_title(
        "Text conditioning via cross-attention (Music Transformer)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b_h = box(ax, 18, 70, 22, 14, "Hidden states\n(music tokens)")
    b_q = box(ax, 42, 70, 12, 12, "Query")
    b_txt = box(
        ax, 18, 28, 24, 16, "English sentence\n(caption / prompt)", face=BOX_FACE_ACCENT
    )
    b_enc = box(
        ax, 48, 28, 24, 16, "MiniLM (freeze)\n+ projection", face=BOX_FACE_ACCENT
    )
    b_kv = box(ax, 72, 28, 16, 14, "Key / Value\n(cond c)", face=BOX_FACE_OK)
    b_ca = box(ax, 72, 70, 16, 14, "Cross-Attention")
    b_out = box(ax, 92, 70, 14, 14, "Hidden\n+ condition", face=BOX_FACE_OK)

    arrow(ax, edge_right(b_h), edge_left(b_q))
    arrow(ax, edge_right(b_q), edge_left(b_ca))
    arrow(ax, edge_right(b_txt), edge_left(b_enc))
    arrow(ax, edge_right(b_enc), edge_left(b_kv))
    arrow(ax, edge_top(b_kv), edge_bot(b_ca))
    arrow(ax, edge_right(b_ca), edge_left(b_out))

    footnote(
        ax,
        "Not 6-ID embeddings · c is a single global vector (length 1 for cross-attn)",
    )
    save(fig, "dieu_Kien_hoa.png")


# ---------------------------------------------------------------------------
# 5. Overall system pipeline
# ---------------------------------------------------------------------------
def gen_system_overview():
    fig, ax = _setup(12, 5.8)
    ax.set_title(
        "Unified I/O: labels / English text → MIDI (two generators)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b1 = box(ax, 12, 75, 18, 16, "6 attributes\n(labels / meta)", face=BOX_FACE_ACCENT)
    b2 = box(ax, 12, 40, 18, 14, "English prompt\n(user / eval)", face=BOX_FACE_ACCENT)
    b3 = box(ax, 36, 58, 20, 18, "Caption builder\nor raw text", face=BOX_FACE_ACCENT)
    b4 = box(
        ax,
        60,
        58,
        20,
        18,
        "TextPromptEncoder\nMiniLM freeze + proj",
        face=BOX_FACE_OK,
    )
    b5 = box(ax, 82, 80, 16, 14, "Music\nTransformer")
    b6 = box(ax, 82, 36, 16, 14, "Piano-roll\nDiffusion")
    b7 = box(ax, 82, 10, 16, 12, "MIDI .mid", face=BOX_FACE_LOSS)

    arrow(ax, edge_right(b1), (26, 75))
    arrow(ax, (26, 75), (26, 58))
    arrow(ax, edge_right(b2), (26, 40))
    arrow(ax, (26, 40), (26, 58))
    arrow(ax, edge_right(b3), edge_left(b4))
    arrow(ax, edge_right(b4), (72, 58))
    arrow(ax, (72, 58), (72, 80))
    arrow(ax, (72, 80), edge_left(b5))
    arrow(ax, (72, 58), (72, 36))
    arrow(ax, (72, 36), edge_left(b6))
    arrow(ax, edge_bot(b5), (82, 24))
    arrow(ax, edge_bot(b6), edge_top(b7))

    footnote(
        ax,
        "Same data split · same text pipeline · compare on MIDI metrics (not CE vs MSE)",
        y=2,
    )
    save(fig, "system_overview_text.png")


# ---------------------------------------------------------------------------
# 6. Sampling loop
# ---------------------------------------------------------------------------
def gen_sampling():
    fig, ax = _setup(7.5, 8.5)
    ax.set_title(
        "Autoregressive sampling (Music Transformer)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=6,
        fontweight="bold",
    )

    b0 = box(ax, 50, 90, 36, 10, "Logits (+ temperature)", face=BOX_FACE)
    b1 = box(
        ax, 50, 74, 40, 12, "Sampling:\ntemperature + top-p (nucleus)", face=BOX_FACE
    )
    b2 = box(ax, 50, 58, 28, 10, "New token", face=BOX_FACE_OK)
    b3 = box(ax, 50, 42, 36, 12, "EOS or max_length?", face=BOX_FACE_ACCENT)
    b_no = box(ax, 22, 24, 22, 12, "No → append\ncontinue", face=BOX_FACE)
    b_yes = box(ax, 78, 24, 24, 12, "Yes → decode\nREMI → MIDI", face=BOX_FACE_LOSS)

    arrow(ax, edge_bot(b0), edge_top(b1))
    arrow(ax, edge_bot(b1), edge_top(b2))
    arrow(ax, edge_bot(b2), edge_top(b3))
    arrow(ax, edge_bot(b3), edge_top(b_no))
    arrow(ax, edge_bot(b3), edge_top(b_yes))
    arrow(ax, edge_left(b_no), (8, 24), rad=0)
    arrow(ax, (8, 24), (8, 90), rad=0)
    arrow(ax, (8, 90), edge_left(b0), rad=0)

    footnote(ax, "Default: τ≈0.85, top-p≈0.9 · KV-cache in generator")
    save(fig, "lay_mau_chuoi.png")


# ---------------------------------------------------------------------------
# 7. CFG (also write safe filename without spaces)
# ---------------------------------------------------------------------------
def gen_cfg():
    fig, ax = _setup(7.5, 6.0)
    ax.set_title(
        "Classifier-Free Guidance (Diffusion)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=6,
        fontweight="bold",
    )

    b_x = box(ax, 50, 88, 16, 10, "x_t")
    b_u = box(ax, 28, 60, 28, 14, "UNet uncond\nc = 0", face=BOX_FACE)
    b_c = box(ax, 72, 60, 28, 14, "UNet cond\nc = text emb", face=BOX_FACE_ACCENT)
    b_g = box(
        ax,
        50,
        32,
        44,
        16,
        "ε̃ = ε₀ + s · (ε_c − ε₀)\ns ≈ 3.5",
        face=BOX_FACE_OK,
    )
    b_o = box(
        ax, 50, 10, 36, 12, "Stronger denoise step toward c", face=BOX_FACE_LOSS
    )

    arrow(ax, edge_bot(b_x), edge_top(b_u))
    arrow(ax, edge_bot(b_x), edge_top(b_c))
    arrow(ax, edge_bot(b_u), edge_top(b_g))
    arrow(ax, edge_bot(b_c), edge_top(b_g))
    arrow(ax, edge_bot(b_g), edge_top(b_o))

    # Keep legacy name used in .tex + safe alias
    save(fig, "classifierfree guidance.png")
    # Re-open is not needed — rewrite same figure to safe name
    fig2, ax2 = _setup(7.5, 6.0)
    ax2.set_title(
        "Classifier-Free Guidance (Diffusion)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=6,
        fontweight="bold",
    )
    b_x = box(ax2, 50, 88, 16, 10, "x_t")
    b_u = box(ax2, 28, 60, 28, 14, "UNet uncond\nc = 0", face=BOX_FACE)
    b_c = box(ax2, 72, 60, 28, 14, "UNet cond\nc = text emb", face=BOX_FACE_ACCENT)
    b_g = box(
        ax2,
        50,
        32,
        44,
        16,
        "ε̃ = ε₀ + s · (ε_c − ε₀)\ns ≈ 3.5",
        face=BOX_FACE_OK,
    )
    b_o = box(
        ax2, 50, 10, 36, 12, "Stronger denoise step toward c", face=BOX_FACE_LOSS
    )
    arrow(ax2, edge_bot(b_x), edge_top(b_u))
    arrow(ax2, edge_bot(b_x), edge_top(b_c))
    arrow(ax2, edge_bot(b_u), edge_top(b_g))
    arrow(ax2, edge_bot(b_c), edge_top(b_g))
    arrow(ax2, edge_bot(b_g), edge_top(b_o))
    save(fig2, "classifierfree_guidance.png")


# ---------------------------------------------------------------------------
# 8. UNet light
# ---------------------------------------------------------------------------
def gen_unet():
    fig, ax = _setup(5.5, 7.5)
    ax.set_title(
        "Conditional UNet (piano-roll)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=6,
        fontweight="bold",
    )
    steps = [
        ("x_t piano-roll noise\n+ time emb + cond FiLM", BOX_FACE_ACCENT),
        ("Downsampling\nConv ResBlocks", BOX_FACE),
        ("Bottleneck\n+ mid self-attention", BOX_FACE),
        ("Upsampling\n+ skip connections", BOX_FACE),
        ("Predicted noise map ε", BOX_FACE_LOSS),
    ]
    nodes = []
    for i, (t, f) in enumerate(steps):
        yy = 88 - i * 16
        b = box(ax, 50, yy, 70, 12, t, face=f)
        nodes.append(b)
        if i:
            arrow(ax, edge_bot(nodes[i - 1]), edge_top(b))
    save(fig, "unet.png")


# ===========================================================================
# REPLACEMENTS for former dark-background / outdated figures
# ===========================================================================

# ---------------------------------------------------------------------------
# he-qua.png — fair comparison consequences (REMI vs roll)
# ---------------------------------------------------------------------------
def gen_he_qua():
    fig, ax = _setup(11.5, 5.8)
    ax.set_title(
        "Fair comparison: same task, different representations",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b_top = box(
        ax,
        50,
        90,
        55,
        12,
        "Shared: data split · English text → MiniLM + proj · MIDI output · MIDI metrics",
        face=BOX_FACE_OK,
        fs=FS_SMALL,
    )

    b_a = box(
        ax,
        25,
        58,
        36,
        22,
        "A · Music Transformer\nREMI event tokens\nCE · autoregressive\nStrong: event order / sparsity",
        face=BOX_FACE_ACCENT,
    )
    b_b = box(
        ax,
        75,
        58,
        36,
        22,
        "B · Piano-roll Diffusion\n88×256 velocity grid\nMSE · DDIM + CFG\nStrong: density / texture",
        face=BOX_FACE_ACCENT,
    )

    b_kpi = box(
        ax,
        50,
        28,
        52,
        16,
        "Compare product KPIs\nMIDI structure · wall-clock · GPU cost · #params",
        face=BOX_FACE,
    )
    b_no = box(
        ax,
        50,
        8,
        58,
        12,
        "Do NOT rank models by raw train loss (CE ≠ MSE)",
        face=BOX_FACE_LOSS,
        fs=FS_SMALL,
    )

    arrow(ax, edge_bot(b_top), edge_top(b_a))
    arrow(ax, edge_bot(b_top), edge_top(b_b))
    arrow(ax, edge_bot(b_a), edge_top(b_kpi))
    arrow(ax, edge_bot(b_b), edge_top(b_kpi))
    arrow(ax, edge_bot(b_kpi), edge_top(b_no))
    save(fig, "he-qua.png")


# ---------------------------------------------------------------------------
# Transformer_decoder.png — full decoder stack
# ---------------------------------------------------------------------------
def gen_transformer_decoder():
    fig, ax = _setup(6.2, 8.2)
    ax.set_title(
        "Music Transformer decoder stack",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=6,
        fontweight="bold",
    )

    steps = [
        ("Token embedding + RoPE", BOX_FACE),
        ("× N DecoderBlock\n(N = 6)", BOX_FACE_ACCENT),
        ("Masked Self-Attn\n(causal, GQA)", BOX_FACE),
        ("Cross-Attn on cond c\n(text · length 1)", BOX_FACE_OK),
        ("SwiGLU FFN + residual", BOX_FACE),
        ("Linear → vocab logits\n(+ weight tying)", BOX_FACE),
        ("p(token next | past, c)", BOX_FACE_LOSS),
    ]
    nodes = []
    for i, (t, f) in enumerate(steps):
        yy = 90 - i * 12.5
        b = box(ax, 55, yy, 55, 10, t, face=f)
        nodes.append(b)
        if i:
            arrow(ax, edge_bot(nodes[i - 1]), edge_top(b))

    b_c = box(
        ax,
        15,
        90 - 3 * 12.5,
        22,
        14,
        "cond c\nMiniLM\n+ proj",
        face=BOX_FACE_OK,
        fs=FS_SMALL,
    )
    arrow(ax, edge_right(b_c), edge_left(nodes[3]))
    footnote(ax, "Teacher forcing at train · AR sampling at inference", y=3)
    save(fig, "Transformer_decoder.png")


# ---------------------------------------------------------------------------
# ddim.png — DDPM train vs DDIM sample
# ---------------------------------------------------------------------------
def gen_ddim():
    fig, ax = _setup(12.0, 4.6)
    ax.set_title(
        "DDPM training vs DDIM sampling (this project)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b0 = box(
        ax,
        18,
        58,
        28,
        28,
        "DDPM train\nT = 1000 steps\ncosine schedule\npredict noise ε",
        face=BOX_FACE,
    )
    b1 = box(
        ax,
        50,
        58,
        26,
        28,
        "Same UNet\nweights\n(EMA shadow\nat eval)",
        face=BOX_FACE_ACCENT,
    )
    b2 = box(
        ax,
        82,
        58,
        28,
        28,
        "DDIM sample\n≈ 80 steps · η = 0\n+ CFG s ≈ 3.5\nfaster, nearly det.",
        face=BOX_FACE_OK,
    )

    arrow(ax, edge_right(b0), edge_left(b1))
    arrow(ax, edge_right(b1), edge_left(b2))
    ax.text(34, 30, "shared train", fontsize=FS_SMALL, color="#555", ha="center")
    ax.text(66, 30, "generate", fontsize=FS_SMALL, color="#555", ha="center")

    footnote(
        ax,
        "Inference uses fewer reverse steps than training noise schedule · still conditional on text emb c",
        y=8,
    )
    save(fig, "ddim.png")


# ---------------------------------------------------------------------------
# gioihanpiano-roll-MIDI.png — piano-roll limits
# ---------------------------------------------------------------------------
def gen_pianoroll_limits():
    fig, ax = _setup(12.0, 5.5)
    ax.set_title(
        "Limits of single-channel piano-roll MIDI (this project)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b0 = box(
        ax,
        50,
        88,
        42,
        12,
        "Piano-roll fixed grid · 88 pitches × 256 frames @ 24 Hz (~10.7 s)",
        face=BOX_FACE_ACCENT,
    )

    items = [
        (18, "Loses fine\nmicro-timing /\nexpression"),
        (40, "Multi-track\nmerged to\n1 channel"),
        (62, "Dense grid\n→ high FLOPs\nsmall batch"),
        (84, "Threshold\n→ MIDI may\nshift onsets"),
    ]
    leaves = []
    for x, txt in items:
        b = box(ax, x, 52, 20, 22, txt, face=BOX_FACE, fs=FS_SMALL)
        leaves.append(b)
        arrow(ax, edge_bot(b0), edge_top(b))

    b_bot = box(
        ax,
        50,
        16,
        58,
        14,
        "Need MIDI structure metrics — not only train MSE",
        face=BOX_FACE_LOSS,
    )
    for b in leaves:
        arrow(ax, edge_bot(b), edge_top(b_bot))

    footnote(
        ax,
        "Program instrument is assigned at decode (PROGRAM_MAP), not predicted by UNet",
        y=3,
    )
    save(fig, "gioihanpiano-roll-MIDI.png")


# ---------------------------------------------------------------------------
# sinhMiDIbangongtrinhchuoi.png — related work: MIDI sequence models
# ---------------------------------------------------------------------------
def gen_related_midi_seq():
    fig, ax = _setup(12.5, 4.2)
    ax.set_title(
        "Related work: MIDI event / language-model line",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b0 = box(ax, 12, 55, 14, 18, "MIDI\ncorpus")
    b1 = box(ax, 32, 55, 18, 18, "Event /\nREMI tokens")
    b2 = box(
        ax,
        56,
        55,
        26,
        22,
        "AR Transformer LM\nMusic Transformer\n(+ relatives)",
        face=BOX_FACE_ACCENT,
    )
    b3 = box(ax, 80, 55, 16, 16, "Generated\nMIDI")
    b4 = box(
        ax,
        50,
        18,
        36,
        14,
        "This thesis branch → GenAI_Transformer\n+ English text cond (MiniLM)",
        face=BOX_FACE_OK,
        fs=FS_SMALL,
    )

    arrow(ax, edge_right(b0), edge_left(b1))
    arrow(ax, edge_right(b1), edge_left(b2))
    arrow(ax, edge_right(b2), edge_left(b3))
    arrow(ax, edge_bot(b2), edge_top(b4))

    footnote(ax, "Scope: symbolic MIDI · not raw audio LM", y=3)
    save(fig, "sinhMiDIbangongtrinhchuoi.png")


# ---------------------------------------------------------------------------
# audio-tu-van-ban.png — related work: text-to-audio (out of scope)
# ---------------------------------------------------------------------------
def gen_related_text_audio():
    fig, ax = _setup(12.0, 4.0)
    ax.set_title(
        "Related work: text-to-audio systems (out of thesis scope)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b0 = box(ax, 14, 55, 18, 18, "Text\nprompt", face=BOX_FACE_ACCENT)
    b1 = box(
        ax,
        48,
        55,
        36,
        24,
        "MusicLM · MusicGen · AudioLDM …\nwaveform / latent audio tokens",
        face=BOX_FACE,
    )
    b2 = box(ax, 84, 55, 18, 18, "Audio\nWAV", face=BOX_FACE_LOSS)

    b3 = box(
        ax,
        50,
        16,
        55,
        14,
        "This project stops at MIDI (editable, cheaper) · not competing on audio MOS",
        face=BOX_FACE_OK,
        fs=FS_SMALL,
    )

    arrow(ax, edge_right(b0), edge_left(b1))
    arrow(ax, edge_right(b1), edge_left(b2))
    ax.annotate(
        "out of scope",
        xy=(66, 38),
        xytext=(66, 38),
        fontsize=FS_SMALL,
        color="#888",
        ha="center",
    )
    arrow(ax, edge_bot(b1), edge_top(b3))

    footnote(ax, "Cited as quality / scale reference only", y=3)
    save(fig, "audio-tu-van-ban.png")


# ---------------------------------------------------------------------------
# diffusion-2d.png — spectrogram diffusion vs piano-roll diffusion
# ---------------------------------------------------------------------------
def gen_related_diffusion_2d():
    fig, ax = _setup(12.0, 5.2)
    ax.set_title(
        "2D diffusion: audio spectrogram vs piano-roll MIDI",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    # Top path: audio
    b_s = box(ax, 16, 78, 20, 14, "Spectrogram\n/ mel")
    b_d1 = box(ax, 48, 78, 22, 14, "Latent / 2D\nDiffusion")
    b_a = box(ax, 80, 78, 16, 14, "Audio", face=BOX_FACE_LOSS)
    arrow(ax, edge_right(b_s), edge_left(b_d1))
    arrow(ax, edge_right(b_d1), edge_left(b_a))
    ax.text(48, 92, "AudioLDM-style (reference)", fontsize=FS_SMALL, color="#555", ha="center")

    # Bottom path: this project
    b_p = box(ax, 16, 38, 20, 14, "Piano-roll\n88×T", face=BOX_FACE_ACCENT)
    b_d2 = box(ax, 48, 38, 22, 14, "UNet Diffusion\nFiLM + CFG", face=BOX_FACE_ACCENT)
    b_m = box(ax, 80, 38, 16, 14, "MIDI", face=BOX_FACE_OK)
    arrow(ax, edge_right(b_p), edge_left(b_d2))
    arrow(ax, edge_right(b_d2), edge_left(b_m))

    b_br = box(
        ax,
        50,
        12,
        40,
        12,
        "This thesis branch → GenAI_Diffusion\n(no VAE / no CLAP)",
        face=BOX_FACE_OK,
        fs=FS_SMALL,
    )
    arrow(ax, edge_bot(b_d2), edge_top(b_br))

    footnote(ax, "Same generative family, different representation & output modality", y=2)
    save(fig, "diffusion-2d.png")


# ---------------------------------------------------------------------------
# Extra: REMI vs piano-roll (fills former figtodo)
# ---------------------------------------------------------------------------
def gen_remi_vs_pianoroll():
    fig, ax = _setup(12.5, 5.5)
    ax.set_title(
        "Same MIDI file → two ML representations",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b_mid = box(ax, 50, 88, 22, 12, "Source MIDI", face=BOX_FACE_OK)

    b_l = box(
        ax,
        25,
        48,
        38,
        36,
        "REMI event stream\n\nNOTE_ON · NOTE_OFF\nTIME_SHIFT · VEL · INST\n\nSilence = time-shift tokens\nDiscrete · sparse events",
        face=BOX_FACE_ACCENT,
        fs=FS_SMALL,
    )
    b_r = box(
        ax,
        75,
        48,
        38,
        36,
        "Piano-roll grid\n\nPitch × time frames\nVelocity in [-1, 1]\n\nTracks merged → 1 channel\nDense · image-like",
        face=BOX_FACE_ACCENT,
        fs=FS_SMALL,
    )

    b_tf = box(ax, 25, 12, 30, 12, "→ Music Transformer", face=BOX_FACE)
    b_df = box(ax, 75, 12, 30, 12, "→ Piano-roll Diffusion", face=BOX_FACE)

    arrow(ax, edge_bot(b_mid), edge_top(b_l))
    arrow(ax, edge_bot(b_mid), edge_top(b_r))
    arrow(ax, edge_bot(b_l), edge_top(b_tf))
    arrow(ax, edge_bot(b_r), edge_top(b_df))

    footnote(
        ax,
        "Representation choice is part of the method difference — not only backbone type",
        y=2,
    )
    save(fig, "fig_ch2_remi_vs_pianoroll.png")


# ---------------------------------------------------------------------------
# Extra: full diffusion process (fills former figtodo)
# ---------------------------------------------------------------------------
def gen_diffusion_process():
    fig, ax = _setup(13.0, 5.0)
    ax.set_title(
        "Conditional diffusion on piano-roll (train & sample)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b0 = box(ax, 12, 62, 16, 18, "x₀ clean\npiano-roll")
    b1 = box(ax, 34, 62, 18, 18, "q(x_t | x₀)\nadd noise")
    b2 = box(ax, 58, 62, 18, 20, "ε_θ(x_t, t, c)\nUNet + FiLM", face=BOX_FACE_ACCENT)
    b3 = box(ax, 82, 62, 18, 18, "DDIM reverse\n≈ 80 steps", face=BOX_FACE_OK)

    b_c = box(
        ax, 58, 22, 28, 16, "c = MiniLM(text)\n+ projection", face=BOX_FACE_OK, fs=FS_SMALL
    )
    b_cfg = box(
        ax,
        82,
        22,
        22,
        16,
        "CFG s≈3.5\nc vs c=0",
        face=BOX_FACE_WARN,
        fs=FS_SMALL,
    )
    b_out = box(ax, 96, 62, 10, 14, "MIDI", face=BOX_FACE_LOSS, fs=FS_SMALL)

    arrow(ax, edge_right(b0), edge_left(b1))
    arrow(ax, edge_right(b1), edge_left(b2))
    arrow(ax, edge_right(b2), edge_left(b3))
    arrow(ax, edge_top(b_c), edge_bot(b2))
    arrow(ax, edge_right(b_cfg), (93, 40))
    arrow(ax, (93, 40), edge_bot(b3))
    arrow(ax, edge_right(b3), edge_left(b_out))

    footnote(
        ax,
        "Train: MSE on ε · Sample: DDIM + CFG · Decode: threshold roll → one Instrument(program)",
        y=4,
    )
    save(fig, "fig_ch2_diffusion_process.png")


# ---------------------------------------------------------------------------
# Extra: decode path for Diffusion
# ---------------------------------------------------------------------------
def gen_diff_decode():
    fig, ax = _setup(12.5, 4.2)
    ax.set_title(
        "Piano-roll → MIDI decode (instrument forced at post-process)",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    b0 = box(ax, 12, 55, 16, 18, "Roll after\nDDIM")
    b1 = box(ax, 34, 55, 18, 18, "Map [-1,1]\n→ [0,1]")
    b2 = box(ax, 56, 55, 20, 20, "Threshold +\nclean short\nactivations", face=BOX_FACE_ACCENT)
    b3 = box(
        ax,
        80,
        55,
        22,
        22,
        "1× Instrument\nprogram ←\nPROGRAM_MAP\n(prompt.instrument)",
        face=BOX_FACE_WARN,
        fs=FS_SMALL,
    )
    b4 = box(ax, 50, 14, 40, 12, "instrument_match = 1.0 is NOT UNet timbre learning", face=BOX_FACE_LOSS, fs=FS_SMALL)

    arrow(ax, edge_right(b0), edge_left(b1))
    arrow(ax, edge_right(b1), edge_left(b2))
    arrow(ax, edge_right(b2), edge_left(b3))
    arrow(ax, edge_bot(b3), (80, 28))
    arrow(ax, (80, 28), edge_right(b4))

    footnote(ax, "Text c still conditions the roll via FiLM/CFG; only GM program is assigned after", y=2)
    save(fig, "fig_ch3_diffusion_decode.png")


# ---------------------------------------------------------------------------
# Extra: data composition simple
# ---------------------------------------------------------------------------
def gen_data_composition():
    fig, ax = _setup(10.0, 5.0)
    ax.set_title(
        "Processed MIDI corpus & fixed split",
        fontsize=FS_TITLE,
        color=TITLE,
        pad=8,
        fontweight="bold",
    )

    # simple stacked-style info boxes instead of pie (no extra deps)
    b_tot = box(ax, 50, 85, 40, 12, "Total processed ≈ 14 232 MIDI files", face=BOX_FACE_OK)
    b1 = box(ax, 22, 55, 28, 18, "ComMU\n≈ 11 144", face=BOX_FACE_ACCENT)
    b2 = box(ax, 50, 55, 24, 18, "MidiCaps\n(subset)", face=BOX_FACE)
    b3 = box(ax, 76, 55, 24, 18, "MAESTRO\n(subset)", face=BOX_FACE)
    b_tr = box(ax, 32, 22, 28, 16, "Train\n12 809", face=BOX_FACE_OK)
    b_va = box(ax, 68, 22, 28, 16, "Val\n1 423", face=BOX_FACE_WARN)

    arrow(ax, edge_bot(b_tot), edge_top(b1))
    arrow(ax, edge_bot(b_tot), edge_top(b2))
    arrow(ax, edge_bot(b_tot), edge_top(b3))
    arrow(ax, edge_bot(b1), edge_top(b_tr))
    arrow(ax, edge_bot(b2), (50, 38))
    arrow(ax, (50, 38), edge_top(b_tr))
    arrow(ax, edge_bot(b3), edge_top(b_va))
    arrow(ax, edge_bot(b2), (60, 38))
    arrow(ax, (60, 38), edge_top(b_va))

    footnote(ax, "compare/split.json fixed once · both models use the same lists", y=4)
    save(fig, "fig_ch3_data_composition.png")


def main():
    print("Generating light-background diagrams (current architecture)…")
    # Core pipelines (already light; regenerate for consistency)
    gen_transformer()
    gen_train_diffusion()
    gen_decoder_block()
    gen_conditioning()
    gen_system_overview()
    gen_sampling()
    gen_cfg()
    gen_unet()

    # Replacements for former dark / outdated figures
    gen_he_qua()
    gen_transformer_decoder()
    gen_ddim()
    gen_pianoroll_limits()
    gen_related_midi_seq()
    gen_related_text_audio()
    gen_related_diffusion_2d()

    # Extra figures for former figtodo slots
    gen_remi_vs_pianoroll()
    gen_diffusion_process()
    gen_diff_decode()
    gen_data_composition()
    print("Done.")


if __name__ == "__main__":
    main()
