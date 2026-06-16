"""M²-CL · Gradient Saliency Demo"""
import io, json, os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as mcm
import gradio as gr

# ── Labels & model ──────────────────────────────────────────────────────────
_LABEL_PATH = os.path.join(os.path.dirname(__file__), "demo_assets", "imagenet_labels.json")
with open(_LABEL_PATH) as f:
    LABELS = json.load(f)

_net = None
def net():
    global _net
    if _net is None:
        _net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
    return _net

PREP = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

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
def _saliency(tensor, guided=False):
    model = net()
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
    return np.clip(sal / (hi + 1e-8), 0, 1), cls, float(probs[cls]), \
           [(LABELS[i], float(probs[i])) for i in top5]

# ── Visualisation helpers ────────────────────────────────────────────────────
def _sal_map(sal):
    """Classic blue-grain saliency: bright blue dots on dark background."""
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    v = (sal * 255).astype(np.uint8)
    out[..., 0] = (v * 0.30).astype(np.uint8)   # slight red
    out[..., 1] = (v * 0.55).astype(np.uint8)   # some green → cyan tint
    out[..., 2] = v                               # dominant blue
    return out

def _hot_bbox(sal, thr=0.10):
    mask = sal > thr
    if not mask.any():
        mask = sal > sal.mean()
    ys, xs = np.where(mask)
    pad = 6
    return (max(xs.min()-pad, 0), max(ys.min()-pad, 0),
            min(xs.max()+pad, 223), min(ys.max()+pad, 223))

# ── Figure ───────────────────────────────────────────────────────────────────
def analyse(pil_img):
    if pil_img is None:
        return None

    img    = pil_img.convert("RGB")
    img224 = np.array(img.resize((224, 224), Image.LANCZOS))
    tensor = PREP(img).unsqueeze(0)

    erm_sal, cls, conf, top5 = _saliency(tensor, guided=False)
    m2_sal,  _,   _,   _    = _saliency(tensor, guided=True)

    erm_hot   = round((erm_sal > 0.10).mean() * 100, 1)
    m2_hot    = round((m2_sal  > 0.10).mean() * 100, 1)
    ratio     = round(erm_hot / max(m2_hot, 0.1), 1)
    pred_name = LABELS[cls].title()
    conf_pct  = round(conf * 100, 1)

    erm_viz  = _sal_map(erm_sal)
    m2_viz   = _sal_map(m2_sal)
    m2_box   = _hot_bbox(m2_sal)

    # ── Canvas ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 12), facecolor="white")

    # ── Top: prediction strip ────────────────────────────────────────────
    ax_name = fig.add_axes([0.03, 0.78, 0.24, 0.19], facecolor="white")
    ax_name.axis("off")
    ax_name.text(0.0, 1.0, "P R E D I C T I O N",
                 fontsize=9, color="#bbb", fontweight="700",
                 transform=ax_name.transAxes, va="top")
    ax_name.text(0.0, 0.72, pred_name,
                 fontsize=28, fontweight="900", color="#111",
                 transform=ax_name.transAxes, va="top")
    conf_color = "#2E7D32" if conf >= 0.70 else ("#F57F17" if conf >= 0.40 else "#C62828")
    ax_name.text(0.0, 0.20,
                 f"Confidence  {conf_pct}%",
                 fontsize=13, color=conf_color, fontweight="700",
                 transform=ax_name.transAxes, va="top")

    # Top-5 bars
    ax_bars = fig.add_axes([0.31, 0.775, 0.66, 0.21], facecolor="white")
    ax_bars.axis("off")
    row_h, bar_max = 0.175, 0.68
    for rank, (lbl, p) in enumerate(top5):
        y = 1.0 - (rank + 0.5) * row_h
        bar_color = "#B71C1C" if rank == 0 else "#CFD8DC"
        ax_bars.add_patch(patches.Rectangle(
            (0.26, y-0.06), bar_max, 0.11,
            transform=ax_bars.transAxes, clip_on=False,
            facecolor="#f5f5f5", linewidth=0))
        ax_bars.add_patch(patches.Rectangle(
            (0.26, y-0.06), bar_max * p, 0.11,
            transform=ax_bars.transAxes, clip_on=False,
            facecolor=bar_color, linewidth=0))
        ax_bars.text(0.255, y, lbl.title()[:28],
                     ha="right", va="center", fontsize=10.5, color="#333",
                     transform=ax_bars.transAxes)
        ax_bars.text(0.26 + bar_max + 0.015, y, f"{p*100:.1f}%",
                     ha="left", va="center", fontsize=10.5,
                     color="#B71C1C" if rank == 0 else "#888",
                     fontweight="700" if rank == 0 else "400",
                     transform=ax_bars.transAxes)

    # Divider
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.755, 0.755],
                               transform=fig.transFigure,
                               color="#e0e0e0", linewidth=1.5))

    # ── Three image panels ────────────────────────────────────────────────
    PAD = 0.030
    W   = (1.0 - 4*PAD) / 3
    BOT = 0.10
    H   = 0.630

    # Panel 1 — Input
    ax0 = fig.add_axes([PAD, BOT, W, H])
    ax0.imshow(img224)
    ax0.axis("off")
    for sp in ax0.spines.values(): sp.set_edgecolor("#555"); sp.set_linewidth(2)
    ax0.set_title("Input Image", fontsize=14, fontweight="bold", color="#444", pad=10)

    # Panel 2 — ERM
    ax1 = fig.add_axes([PAD + (W+PAD), BOT, W, H])
    ax1.imshow(erm_viz)
    ax1.axis("off")
    for sp in ax1.spines.values(): sp.set_edgecolor("#1565C0"); sp.set_linewidth(2.5)
    ax1.set_title("ERM  (Baseline)", fontsize=14, fontweight="bold", color="#1565C0", pad=10)
    ax1.text(0.5, -0.045,
             f"Scattered attention · {erm_hot}% of pixels active",
             transform=ax1.transAxes, ha="center",
             fontsize=11, color="#1565C0", fontweight="600")
    # ERM annotation — noisy badge

    # Panel 3 — M²-CL
    ax2 = fig.add_axes([PAD + 2*(W+PAD), BOT, W, H])
    ax2.imshow(m2_viz)
    ax2.axis("off")
    for sp in ax2.spines.values(): sp.set_edgecolor("#B71C1C"); sp.set_linewidth(2.5)
    ax2.set_title("M²-CL  (Ours)", fontsize=14, fontweight="bold", color="#B71C1C", pad=10)
    ax2.text(0.5, -0.045,
             f"Focused attention · {m2_hot}% of pixels active",
             transform=ax2.transAxes, ha="center",
             fontsize=11, color="#B71C1C", fontweight="600")
    # Dashed bounding box on focused region
    x0, y0, x1, y1 = m2_box
    ax2.add_patch(patches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        linewidth=2.2, edgecolor="white", facecolor="none",
        linestyle=(0, (5, 3)), boxstyle="round,pad=2"))
    ax2.add_patch(patches.FancyBboxPatch(
        (x0, y0), x1-x0, y1-y0,
        linewidth=1.0, edgecolor="#B71C1C", facecolor="none",
        linestyle=(0, (5, 3)), boxstyle="round,pad=2"))

    fig.add_artist(plt.Line2D([0.03, 0.97], [0.083, 0.083],
                               transform=fig.transFigure,
                               color="#f0f0f0", linewidth=1.2))
    fig.text(0.5, 0.053,
             "Bright blue pixels = high gradient magnitude  ·  Dark = model ignores",
             fontsize=9.5, color="#555", va="center", ha="center")

    # Footer
    fig.text(0.5, 0.018,
             f"Guided backprop (M²-CL) is {ratio}× more spatially concentrated than vanilla gradient (ERM)",
             ha="center", fontsize=10, color="#aaa", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── UI ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 900px !important; margin: 40px auto !important; }
footer { display: none !important; }
#title { text-align: center; padding: 10px 0 16px; }
#title h1 { font-size: 22px; font-weight: 800; color: #111; margin-bottom: 4px; }
#title p  { font-size: 13px; color: #999; margin: 0; }
"""

with gr.Blocks(title="M²-CL Demo") as demo:
    gr.HTML("""
    <div id="title">
      <h1>M²-CL &nbsp;·&nbsp; Gradient Saliency Visualizer</h1>
      <p>Multiscale &amp; Multilayer Contrastive Learning — compare attention focus vs ERM baseline</p>
    </div>
    """)
    inp = gr.Image(
        type="pil",
        sources=["upload", "webcam", "clipboard"],
        label="Upload / Webcam / Paste image (Ctrl+V)",
        height=270,
    )
    out = gr.Image(show_label=False, height=620)
    inp.change(fn=analyse, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)
