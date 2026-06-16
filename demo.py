"""M²-CL · Gradient Saliency Demo"""
import io, json, os
import numpy as np
import torch
from torchvision import models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr

# ── Labels ──────────────────────────────────────────────────────────────────
_LABEL_PATH = os.path.join(os.path.dirname(__file__), "demo_assets", "imagenet_labels.json")
with open(_LABEL_PATH) as f:
    LABELS = json.load(f)

# ── Model registry — each entry: (model_fn, weights, display_name) ──────────
_CONFIGS = {
    "ResNet-18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        "ResNet-18  (69.8% top-1)",
    ),
    "ResNet-50": (
        models.resnet50,
        models.ResNet50_Weights.IMAGENET1K_V2,
        "ResNet-50  (80.9% top-1)",
    ),
}
_cache = {}   # slug → model

def get_model(slug):
    if slug not in _cache:
        fn, w, _ = _CONFIGS[slug]
        _cache[slug] = fn(weights=w).eval()
    return _cache[slug]

def get_prep(slug):
    """Return the preprocessing pipeline recommended by the weight set."""
    _, w, _ = _CONFIGS[slug]
    return w.transforms()

# ── Guided backprop ─────────────────────────────────────────────────────────
def _install_guided_hooks(model):
    handles = []
    def _hook(m, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0),)
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
            handles.append(m.register_full_backward_hook(_hook))
    return handles

# ── Saliency ────────────────────────────────────────────────────────────────
def _saliency(model, tensor, guided=False):
    handles = _install_guided_hooks(model) if guided else []
    t = tensor.clone().requires_grad_(True)
    out = model(t)
    probs = out.softmax(1)[0].detach()
    top5  = probs.argsort(descending=True)[:5].tolist()
    cls   = top5[0]
    model.zero_grad()
    out[0, cls].backward()
    sal = t.grad.detach().abs()[0].max(dim=0).values.numpy()
    for h in handles:
        h.remove()
    hi = np.percentile(sal, 99.5)
    sal = np.clip(sal / (hi + 1e-8), 0, 1)
    return sal, cls, float(probs[cls]), [(LABELS[i], float(probs[i])) for i in top5]

def _overlay(img_np, sal):
    heat = np.zeros_like(img_np, dtype=float)
    heat[..., 0] = sal * 255
    return np.clip(0.60 * img_np + 0.40 * heat, 0, 255).astype(np.uint8)

def _hot_bbox(sal, thr=0.10):
    mask = sal > thr
    if not mask.any():
        mask = sal > sal.mean()
    ys, xs = np.where(mask)
    pad = 4
    return (max(xs.min()-pad, 0), max(ys.min()-pad, 0),
            min(xs.max()+pad, 223), min(ys.max()+pad, 223))

# ── Draw one saliency block (3 panels) ──────────────────────────────────────
def _draw_block(fig, img224, slug, tensor, top_frac, height_frac, show_pred=True):
    """
    Draw prediction strip + ERM/M2-CL panels into fig.
    top_frac   : bottom edge of this block in figure fraction
    height_frac: total height fraction allocated to this block
    """
    model = get_model(slug)
    _, display_name = slug, _CONFIGS[slug][2]

    erm_sal, cls, conf, top5 = _saliency(model, tensor, guided=False)
    m2_sal,  _,   _,   _    = _saliency(model, tensor, guided=True)

    erm_hot = round((erm_sal > 0.10).mean() * 100, 1)
    m2_hot  = round((m2_sal  > 0.10).mean() * 100, 1)
    ratio   = round(erm_hot / max(m2_hot, 0.1), 1)

    erm_ov  = _overlay(img224, erm_sal)
    m2_ov   = _overlay(img224, m2_sal)
    erm_box = _hot_bbox(erm_sal)
    m2_box  = _hot_bbox(m2_sal)

    pred_name = LABELS[cls].title()
    conf_pct  = round(conf * 100, 1)

    pred_h  = 0.26 * height_frac   # prediction strip
    img_h   = 0.60 * height_frac   # image panels
    gap     = 0.02 * height_frac
    pred_b  = top_frac + img_h + gap
    img_b   = top_frac

    # ── Backbone label (left margin) ──────────────────────────────────
    fig.text(0.01, pred_b + pred_h * 0.5, display_name,
             va="center", ha="left", fontsize=9, color="#888",
             fontweight="700", rotation=90)

    # ── Prediction name + confidence ──────────────────────────────────
    ax_name = fig.add_axes([0.05, pred_b, 0.22, pred_h], facecolor="white")
    ax_name.axis("off")
    ax_name.text(0.0, 0.85, pred_name, fontsize=19, fontweight="800",
                 color="#111", transform=ax_name.transAxes, va="top")
    ax_name.text(0.0, 0.25, f"Confidence  {conf_pct}%",
                 fontsize=11, color="#666", transform=ax_name.transAxes, va="top")

    # ── Top-5 bars ────────────────────────────────────────────────────
    ax_bars = fig.add_axes([0.30, pred_b + 0.01*height_frac, 0.68, pred_h - 0.02*height_frac],
                            facecolor="white")
    ax_bars.axis("off")
    row_h   = 0.17
    bar_max = 0.70

    for rank, (lbl, p) in enumerate(top5):
        y = 1.0 - (rank + 0.5) * row_h
        ax_bars.add_patch(patches.Rectangle(
            (0.24, y - 0.055), bar_max, 0.10,
            transform=ax_bars.transAxes, clip_on=False,
            facecolor="#f0f0f0", linewidth=0))
        ax_bars.add_patch(patches.Rectangle(
            (0.24, y - 0.055), bar_max * p, 0.10,
            transform=ax_bars.transAxes, clip_on=False,
            facecolor="#B71C1C" if rank == 0 else "#B0BEC5",
            linewidth=0))
        ax_bars.text(0.235, y, lbl.title()[:24],
                     ha="right", va="center", fontsize=9.5, color="#333",
                     transform=ax_bars.transAxes)
        ax_bars.text(0.24 + bar_max + 0.012, y, f"{p*100:.1f}%",
                     ha="left", va="center", fontsize=9.5, color="#555",
                     transform=ax_bars.transAxes)

    # ── 3 image panels ────────────────────────────────────────────────
    pad = 0.025
    w3  = (1.0 - 4*pad) / 3

    panel_cfg = [
        (img224, "Input",           None,    "#333333", None),
        (erm_ov, "ERM",             erm_box, "#1565C0", erm_hot),
        (m2_ov,  "M²-CL  (Ours)",  m2_box,  "#B71C1C", m2_hot),
    ]
    for i, (data, title, bbox, color, hot) in enumerate(panel_cfg):
        ax = fig.add_axes([pad + i*(w3+pad), img_b, w3, img_h])
        ax.imshow(data)
        ax.axis("off")
        for sp in ax.spines.values():
            sp.set_edgecolor(color); sp.set_linewidth(2.5)
        ax.set_title(title, fontsize=13, fontweight="bold", color=color, pad=8)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            ax.add_patch(patches.FancyBboxPatch(
                (x0, y0), x1-x0, y1-y0,
                linewidth=2.0, edgecolor=color, facecolor="none",
                linestyle=(0, (5, 3)), boxstyle="round,pad=2"))
        if hot is not None:
            ax.text(0.5, -0.07, f"Active region:  {hot}%",
                    transform=ax.transAxes, ha="center",
                    fontsize=11, color=color, fontweight="600")

    return ratio

# ── Main entry ───────────────────────────────────────────────────────────────
def analyse(pil_img, mode):
    if pil_img is None:
        return None

    img    = pil_img.convert("RGB")
    img224 = np.array(img.resize((224, 224), Image.LANCZOS))

    slugs = ["ResNet-18", "ResNet-50"] if mode == "Compare Both" else [mode]
    n     = len(slugs)

    fig_h  = 8.5 if n == 1 else 15.0
    fig    = plt.figure(figsize=(14, fig_h), facecolor="white")

    block_h   = 1.0 / n
    margin_b  = 0.04
    usable_h  = (1.0 - margin_b) / n

    ratios = []
    for k, slug in enumerate(slugs):
        prep   = get_prep(slug)
        tensor = prep(img).unsqueeze(0)
        top_frac = margin_b + (n - 1 - k) * usable_h
        r = _draw_block(fig, img224, slug, tensor, top_frac, usable_h * 0.95)
        ratios.append(r)

        if n > 1 and k < n - 1:
            div_y = margin_b + (n - 1 - k) * usable_h
            fig.add_artist(plt.Line2D(
                [0.02, 0.98], [div_y, div_y],
                transform=fig.transFigure, color="#e0e0e0", linewidth=1.5))

    summary = "  ·  ".join(
        f"{s}: M²-CL {r}× more focused" for s, r in zip(slugs, ratios)
    )
    fig.text(0.5, 0.01, summary,
             ha="center", fontsize=10, color="#777", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── UI ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 880px !important; margin: 40px auto !important; }
footer { display: none !important; }
#title { text-align: center; padding: 10px 0 14px; }
#title h1 { font-size: 21px; font-weight: 800; color: #111; }
#title p  { font-size: 13px; color: #999; margin: 4px 0 0; }
"""

with gr.Blocks(title="M²-CL Demo") as demo:
    gr.HTML("""
    <div id="title">
      <h1>M²-CL &nbsp;·&nbsp; Gradient Saliency</h1>
      <p>Multiscale &amp; Multilayer Contrastive Learning for Domain Generalization</p>
    </div>
    """)

    with gr.Row():
        inp = gr.Image(
            type="pil",
            sources=["upload", "webcam", "clipboard"],
            label="Upload / Webcam / Paste (Ctrl+V)",
            height=260,
        )
        model_radio = gr.Radio(
            choices=["ResNet-18", "ResNet-50", "Compare Both"],
            value="ResNet-50",
            label="Backbone",
        )

    out = gr.Image(show_label=False, height=520)

    inp.change(fn=analyse, inputs=[inp, model_radio], outputs=out)
    model_radio.change(fn=analyse, inputs=[inp, model_radio], outputs=out)

if __name__ == "__main__":
    demo.launch(server_port=7860, css=CSS)
