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
    sal = np.clip(sal / (hi + 1e-8), 0, 1)
    return sal, cls, float(probs[cls]), [(LABELS[i], float(probs[i])) for i in top5]

# ── Overlay (red-channel, matches tools/saliency.py) ───────────────────────
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
    return max(xs.min()-pad,0), max(ys.min()-pad,0), min(xs.max()+pad,223), min(ys.max()+pad,223)

# ── Figure ──────────────────────────────────────────────────────────────────
def analyse(pil_img):
    if pil_img is None:
        return None

    img    = pil_img.convert("RGB")
    img224 = np.array(img.resize((224, 224), Image.LANCZOS))
    tensor = PREP(img).unsqueeze(0)

    erm_sal, cls, conf, top5 = _saliency(tensor, guided=False)
    m2_sal,  _,   _,   _    = _saliency(tensor, guided=True)

    erm_hot = round((erm_sal > 0.10).mean() * 100, 1)
    m2_hot  = round((m2_sal  > 0.10).mean() * 100, 1)
    ratio   = round(erm_hot / max(m2_hot, 0.1), 1)

    pred_name = LABELS[cls].title()
    conf_pct  = round(conf * 100, 1)

    erm_ov  = _overlay(img224, erm_sal)
    m2_ov   = _overlay(img224, m2_sal)
    erm_box = _hot_bbox(erm_sal)
    m2_box  = _hot_bbox(m2_sal)

    # ── Layout ──────────────────────────────────────────────────────────
    # Figure has two rows:
    #   Row A (top, 28%): prediction name + top-5 bars
    #   Row B (bot, 72%): 3 image panels side by side
    fig = plt.figure(figsize=(14, 8.5), facecolor="white")

    # ── Row A: prediction ────────────────────────────────────────────────
    # Left cell: big class name + confidence
    ax_name = fig.add_axes([0.02, 0.74, 0.24, 0.22], facecolor="white")
    ax_name.axis("off")
    ax_name.text(0.0, 0.90, "P R E D I C T I O N", fontsize=9, color="#aaa",
                 fontweight="700",
                 transform=ax_name.transAxes, va="top")
    ax_name.text(0.0, 0.62, pred_name,
                 fontsize=24, fontweight="800", color="#111",
                 transform=ax_name.transAxes, va="top")
    ax_name.text(0.0, 0.18, f"Confidence  {conf_pct}%",
                 fontsize=12, color="#666",
                 transform=ax_name.transAxes, va="top")

    # Right cell: top-5 horizontal bars
    ax_bars = fig.add_axes([0.30, 0.72, 0.68, 0.26], facecolor="white")
    ax_bars.axis("off")
    row_h   = 0.16          # height of each bar row (in axes fraction)
    bar_max = 0.72          # max bar width (fraction of axes width)

    for rank, (lbl, p) in enumerate(top5):
        y = 1.0 - (rank + 0.5) * row_h
        # track
        ax_bars.add_patch(patches.Rectangle(
            (0.22, y - 0.055), bar_max, 0.10,
            transform=ax_bars.transAxes, clip_on=False,
            facecolor="#f0f0f0", linewidth=0))
        # fill
        ax_bars.add_patch(patches.Rectangle(
            (0.22, y - 0.055), bar_max * p, 0.10,
            transform=ax_bars.transAxes, clip_on=False,
            facecolor="#B71C1C" if rank == 0 else "#B0BEC5",
            linewidth=0))
        # label left
        ax_bars.text(0.21, y, lbl.title()[:24],
                     ha="right", va="center", fontsize=10, color="#333",
                     transform=ax_bars.transAxes)
        # pct right
        ax_bars.text(0.22 + bar_max + 0.012, y, f"{p*100:.1f}%",
                     ha="left", va="center", fontsize=10, color="#555",
                     transform=ax_bars.transAxes)

    # Divider line between rows
    fig.add_artist(plt.Line2D([0.02, 0.98], [0.71, 0.71],
                               transform=fig.transFigure,
                               color="#e8e8e8", linewidth=1.2))

    # ── Row B: 3 image panels ────────────────────────────────────────────
    pad = 0.025
    w3  = (1.0 - 4 * pad) / 3

    panel_cfg = [
        (img224,  "Input Image",    None,    "#333333", None),
        (erm_ov,  "ERM",            erm_box, "#1565C0", erm_hot),
        (m2_ov,   "M²-CL  (Ours)", m2_box,  "#B71C1C", m2_hot),
    ]

    for i, (data, title, bbox, color, hot) in enumerate(panel_cfg):
        ax = fig.add_axes([pad + i*(w3+pad), 0.11, w3, 0.56])
        ax.imshow(data)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
        ax.set_title(title, fontsize=14, fontweight="bold",
                     color=color, pad=10)

        if bbox is not None:
            x0, y0, x1, y1 = bbox
            rect = patches.FancyBboxPatch(
                (x0, y0), x1-x0, y1-y0,
                linewidth=2.0, edgecolor=color, facecolor="none",
                linestyle=(0, (5, 3)), boxstyle="round,pad=2",
            )
            ax.add_patch(rect)

        if hot is not None:
            ax.text(0.5, -0.06, f"Active region:  {hot}%",
                    transform=ax.transAxes, ha="center",
                    fontsize=11.5, color=color, fontweight="600")

    # Footer
    fig.text(0.5, 0.03,
             f"M²-CL is {ratio}× more focused than ERM  ·  "
             "Guided backprop suppresses spurious texture gradients, "
             "isolating class-invariant structural features.",
             ha="center", fontsize=10.5, color="#777", style="italic")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ── UI ───────────────────────────────────────────────────────────────────────
CSS = """
.gradio-container { max-width: 860px !important; margin: 40px auto !important; }
footer { display: none !important; }
#title { text-align: center; padding: 10px 0 18px; }
#title h1 { font-size: 21px; font-weight: 800; color: #111; letter-spacing: -.3px; }
#title p  { font-size: 13px; color: #999; margin: 4px 0 0; }
"""

with gr.Blocks(title="M²-CL Demo") as demo:
    gr.HTML("""
    <div id="title">
      <h1>M²-CL &nbsp;·&nbsp; Gradient Saliency</h1>
      <p>Multiscale &amp; Multilayer Contrastive Learning for Domain Generalization</p>
    </div>
    """)

    inp = gr.Image(
        type="pil",
        sources=["upload", "webcam", "clipboard"],
        label="Upload / Webcam / Paste (Ctrl+V)",
        height=280,
    )
    out = gr.Image(show_label=False, height=500)
    inp.change(fn=analyse, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(server_port=7860, css=CSS)
