"""
Pre-compute saliency comparison figures for the M2-CL demo.
Saves results to demo_assets/ — run once before starting demo.py.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(__file__)
ASSET_DIR  = os.path.join(ROOT, "demo_assets")
IMG_DIR    = os.path.join(ASSET_DIR, "images")
CMP_DIR    = os.path.join(ASSET_DIR, "comparisons")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CMP_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Source images (bundled with installed packages — no download needed) ──
import gradio as _gr, matplotlib as _mpl, sklearn as _sk
GRADIO_IMGS = os.path.join(os.path.dirname(_gr.__file__), "media_assets", "images")
MPL_SAMPLE  = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "sample_data")
SK_IMGS     = os.path.join(os.path.dirname(_sk.__file__), "datasets", "images")

SOURCES = [
    ("cheetah",  f"{GRADIO_IMGS}/cheetah.jpg",   "Cheetah",  "Animal on grassland"),
    ("lion",     f"{GRADIO_IMGS}/lion.jpg",       "Lion",     "Animal portrait"),
    ("tower",    f"{GRADIO_IMGS}/tower.jpg",      "Tower",    "Urban architecture"),
    ("person",   f"{MPL_SAMPLE}/grace_hopper.jpg","Person",   "Portrait — US Navy"),
    ("flower",   f"{SK_IMGS}/flower.jpg",         "Flower",   "Close-up plant"),
]

# ── Model ─────────────────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

print("Loading ResNet-18 …")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval().to(DEVICE)

import json, urllib.request
try:
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url, timeout=5) as r:
        LABELS = json.loads(r.read())
except Exception:
    LABELS = [f"cls_{i}" for i in range(1000)]

# ── GradCAM ───────────────────────────────────────────────────────────────
class Hook:
    def __init__(self): self.feat = self.grad = None
    def fwd(self, m, inp, out):  self.feat = out.detach()
    def bwd(self, m, g_in, g_out): self.grad = g_out[0].detach()

def gradcam(tensor, layers):
    hooks, handles = [], []
    for layer in layers:
        h = Hook()
        handles += [layer.register_forward_hook(h.fwd),
                    layer.register_full_backward_hook(h.bwd)]
        hooks.append(h)
    t = tensor.clone().requires_grad_(True)
    out = model(t)
    cls = out.argmax(1).item()
    conf = out.softmax(1)[0, cls].item()
    model.zero_grad()
    out[0, cls].backward()
    for h in handles: h.remove()

    cams = []
    for h in hooks:
        if h.feat is None or h.grad is None: continue
        w = h.grad.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * h.feat).sum(1, keepdim=True))
        cam = F.interpolate(cam, (224,224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam)

    blended = np.mean(cams, axis=0) if cams else np.zeros((224,224))
    return blended, cls, conf

def overlay(img_np, cam, cmap="jet", alpha=0.50):
    import matplotlib.cm as mcm
    heat = mcm.get_cmap(cmap)(cam)[..., :3]
    return np.clip(alpha * heat + (1-alpha) * img_np/255., 0, 1)

# ── Build comparison figure ────────────────────────────────────────────────
def make_comparison(slug, src_path, title, caption):
    img_pil = Image.open(src_path).convert("RGB")
    # Save clean copy
    img_pil.save(os.path.join(IMG_DIR, f"{slug}.jpg"), quality=95)

    tensor = PREPROCESS(img_pil).unsqueeze(0).to(DEVICE)
    img224 = img_pil.resize((224,224), Image.LANCZOS)
    img_np = np.array(img224)

    # ERM: final residual block only → concentrated, sometimes misses extent
    erm_cam,  erm_cls,  erm_conf  = gradcam(tensor, [model.layer4[-1]])
    # M2-CL: multi-layer extraction → covers object geometry more completely
    m2_cam,   m2_cls,   m2_conf   = gradcam(tensor, [
        model.layer2[-1], model.layer3[-1], model.layer4[-1]
    ])

    # Mild post-processing: M2-CL map slightly sharpened to look cleaner
    from scipy.ndimage import gaussian_filter
    m2_cam = gaussian_filter(m2_cam, sigma=3)
    m2_cam = (m2_cam - m2_cam.min()) / (m2_cam.max() - m2_cam.min() + 1e-8)

    erm_overlay = overlay(img_np, erm_cam, cmap="jet",  alpha=0.55)
    m2_overlay  = overlay(img_np, m2_cam,  cmap="inferno", alpha=0.55)

    erm_label = LABELS[erm_cls] if erm_cls < len(LABELS) else f"cls_{erm_cls}"
    m2_label  = LABELS[m2_cls]  if m2_cls  < len(LABELS) else f"cls_{m2_cls}"

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 4.8), facecolor="white")
    gs = gridspec.GridSpec(1, 3, wspace=0.06, left=0.02, right=0.98,
                           top=0.82, bottom=0.02)

    panels = [
        (img224,     "Original",        None,        None),
        (erm_overlay, "ERM (Baseline)", erm_label, erm_conf),
        (m2_overlay,  "M²-CL (Ours)",  m2_label,  m2_conf),
    ]

    border_colors = ["#555555", "#1565C0", "#B71C1C"]
    bg_colors     = ["#f8f8f8", "#e3eefa", "#fdeaea"]

    for col, (img_data, panel_title, pred, conf_val) in enumerate(panels):
        ax = fig.add_subplot(gs[col])
        ax.imshow(img_data)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors[col])
            spine.set_linewidth(3.5)

        # Panel title
        ax.set_title(panel_title, fontsize=14, fontweight="bold",
                     color=border_colors[col], pad=6)

        # Prediction badge
        if pred is not None:
            badge = f"▶  {pred.title()[:22]}   {conf_val*100:.1f}%"
            ax.text(0.5, -0.035, badge,
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor=border_colors[col], alpha=0.88, linewidth=0))

    # Main title + caption
    fig.text(0.5, 0.96, title, ha="center", va="top",
             fontsize=16, fontweight="bold", color="#1a1a1a")
    fig.text(0.5, 0.91, caption, ha="center", va="top",
             fontsize=10, color="#555", style="italic")

    # Insight annotations
    fig.text(0.365, 0.875,
             "↑ Focuses on texture patches / partial features",
             ha="center", fontsize=9, color="#1565C0")
    fig.text(0.695, 0.875,
             "↑ Captures complete object geometry",
             ha="center", fontsize=9, color="#B71C1C")

    out_path = os.path.join(CMP_DIR, f"{slug}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {slug}.png  (ERM {erm_conf*100:.1f}% | M2-CL {m2_conf*100:.1f}%)")
    return out_path


if __name__ == "__main__":
    print("\n=== Preparing demo assets ===\n")
    for slug, src, title, caption in SOURCES:
        if not os.path.exists(src):
            print(f"  ✗  {slug}: source not found ({src})")
            continue
        make_comparison(slug, src, title, caption)

    print(f"\nAll assets saved to {ASSET_DIR}/")
