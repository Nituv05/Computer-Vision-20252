"""
Generate M2-CL presentation slides using python-pptx.
Run: python slides/generate_slides.py
Output: m2cl_presentation.pptx
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Color palette ──────────────────────────────────────────────
DARK_BLUE  = RGBColor(0x1A, 0x37, 0x6C)
MID_BLUE   = RGBColor(0x2E, 0x75, 0xB6)
LIGHT_BLUE = RGBColor(0xBD, 0xD7, 0xEE)
ORANGE     = RGBColor(0xED, 0x7D, 0x31)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY  = RGBColor(0x40, 0x40, 0x40)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
GREEN      = RGBColor(0x70, 0xAD, 0x47)

W = Inches(13.33)
H = Inches(7.5)


def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    layout = prs.slide_layouts[6]  # completely blank
    return prs.slides.add_slide(layout)


def fill_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, l, t, w, h, fill_color=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, l, t, w, h)  # MSO_SHAPE_TYPE.RECTANGLE
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, text, l, t, w, h,
                 font_size=Pt(18), bold=False, color=DARK_GRAY,
                 align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = font_size
    run.font.bold = bold
    run.font.color.rgb = color
    return txb


def add_header_bar(slide, title, subtitle=None):
    add_rect(slide, 0, 0, W, Inches(1.25), fill_color=DARK_BLUE)
    add_text_box(slide, title,
                 Inches(0.4), Inches(0.1), Inches(12), Inches(0.7),
                 font_size=Pt(28), bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text_box(slide, subtitle,
                     Inches(0.4), Inches(0.78), Inches(12), Inches(0.4),
                     font_size=Pt(14), color=LIGHT_BLUE, align=PP_ALIGN.LEFT)


def add_footer(slide, slide_num, total=15):
    add_rect(slide, 0, H - Inches(0.35), W, Inches(0.35), fill_color=DARK_BLUE)
    add_text_box(slide, "M²-CL | Multi-Scale & Multi-Layer Contrastive Learning",
                 Inches(0.3), H - Inches(0.34), Inches(10), Inches(0.3),
                 font_size=Pt(9), color=LIGHT_BLUE)
    add_text_box(slide, f"{slide_num} / {total}",
                 Inches(12.3), H - Inches(0.34), Inches(0.8), Inches(0.3),
                 font_size=Pt(9), color=WHITE, align=PP_ALIGN.RIGHT)


def add_bullet(slide, items, l, t, w, h, font_size=Pt(16), color=DARK_GRAY, indent=0):
    txb = slide.shapes.add_textbox(l, t, w, h)
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        p.level = indent
        run = p.add_run()
        run.text = item
        run.font.size = font_size
        run.font.color.rgb = color


# ─────────────────────────────────────────────────────────────
# SLIDE BUILDERS
# ─────────────────────────────────────────────────────────────

def slide_01_title(prs):
    s = blank_slide(prs)
    fill_bg(s, DARK_BLUE)

    # Decorative accent
    add_rect(s, 0, Inches(3.8), Inches(0.12), Inches(1.8), fill_color=ORANGE)
    add_rect(s, Inches(0.12), Inches(3.8), Inches(5), Inches(0.06), fill_color=ORANGE)

    add_text_box(s, "M²-CL",
                 Inches(0.5), Inches(1.2), Inches(12), Inches(1.2),
                 font_size=Pt(54), bold=True, color=WHITE, align=PP_ALIGN.LEFT)

    add_text_box(s, "Multi-Scale and Multi-Layer Contrastive Learning\nfor Domain Generalization",
                 Inches(0.5), Inches(2.5), Inches(10), Inches(1.2),
                 font_size=Pt(26), bold=False, color=LIGHT_BLUE, align=PP_ALIGN.LEFT)

    add_text_box(s, "Aristotelis Ballas  ·  Christos Diou",
                 Inches(0.5), Inches(3.9), Inches(10), Inches(0.5),
                 font_size=Pt(18), color=WHITE)

    add_text_box(s, "IEEE Transactions on Artificial Intelligence  ·  2024",
                 Inches(0.5), Inches(4.5), Inches(10), Inches(0.4),
                 font_size=Pt(14), color=LIGHT_BLUE)

    add_text_box(s, "arXiv: 2308.14418",
                 Inches(0.5), Inches(5.0), Inches(5), Inches(0.4),
                 font_size=Pt(13), color=ORANGE)

    add_footer(s, 1)
    return s


def slide_02_outline(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Outline")

    sections = [
        ("1", "Problem Statement", "Domain Generalization in image classification"),
        ("2", "Motivation",        "Why multi-scale and multi-layer representations?"),
        ("3", "Method Overview",   "M²-CL architecture at a glance"),
        ("4", "Extraction Block",  "Channel reduction + spatial dropout + pooling"),
        ("5", "Contrastive Loss",  "Multi-level objective function"),
        ("6", "Training Setup",    "Hyperparameters & DomainBed protocol"),
        ("7", "Datasets",          "PACS · VLCS · Office-Home · NICO"),
        ("8", "Results",           "State-of-the-art on all four benchmarks"),
        ("9", "Ablations",         "What each component contributes"),
        ("10", "Conclusion",        "Key takeaways"),
    ]

    col_l = Inches(0.5)
    for i, (num, title, desc) in enumerate(sections):
        top = Inches(1.4) + i * Inches(0.52)
        add_rect(s, col_l, top, Inches(0.42), Inches(0.38),
                 fill_color=MID_BLUE)
        add_text_box(s, num,
                     col_l + Inches(0.04), top,
                     Inches(0.38), Inches(0.38),
                     font_size=Pt(13), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(s, title,
                     col_l + Inches(0.55), top,
                     Inches(3.5), Inches(0.38),
                     font_size=Pt(14), bold=True, color=DARK_BLUE)
        add_text_box(s, desc,
                     col_l + Inches(4.1), top,
                     Inches(8.5), Inches(0.38),
                     font_size=Pt(13), color=DARK_GRAY)

    add_footer(s, 2)
    return s


def slide_03_problem(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Problem: Domain Generalization",
                   "Train on multiple source domains → generalize to unseen target domain")

    add_text_box(s, "Challenge",
                 Inches(0.5), Inches(1.5), Inches(5.5), Inches(0.45),
                 font_size=Pt(18), bold=True, color=MID_BLUE)

    items = [
        "• Deep CNNs overfit to spurious domain-specific cues",
        "• Distribution shift between train and test domains",
        "• No access to target domain data during training",
        "• Real-world need: medical imaging, autonomous driving, …",
    ]
    add_bullet(s, items, Inches(0.5), Inches(2.0), Inches(5.8), Inches(2.5),
               font_size=Pt(16))

    # Diagram box
    add_rect(s, Inches(7.0), Inches(1.5), Inches(5.8), Inches(5.5),
             fill_color=LIGHT_BLUE, line_color=MID_BLUE, line_width=Pt(1.5))

    add_text_box(s, "Domain Generalization Setup",
                 Inches(7.2), Inches(1.6), Inches(5.4), Inches(0.5),
                 font_size=Pt(14), bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

    domains_diagram = (
        "Source Domains\n"
        "┌─────────────────────────┐\n"
        "│  D₁: Photo              │\n"
        "│  D₂: Art Painting       │\n"
        "│  D₃: Cartoon            │\n"
        "└─────────────────────────┘\n"
        "           │  Train\n"
        "           ▼\n"
        "     ┌───────────┐\n"
        "     │   Model   │\n"
        "     └───────────┘\n"
        "           │  Evaluate\n"
        "           ▼\n"
        "┌─────────────────────────┐\n"
        "│  D₄: Sketch (unseen)    │\n"
        "└─────────────────────────┘"
    )
    add_text_box(s, domains_diagram,
                 Inches(7.2), Inches(2.2), Inches(5.5), Inches(4.5),
                 font_size=Pt(11), color=DARK_BLUE)

    add_footer(s, 3)
    return s


def slide_04_motivation(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Motivation: Multi-Scale & Multi-Layer Features",
                   "Different network depths encode different levels of domain-invariant information")

    add_text_box(s, "Key Insight",
                 Inches(0.5), Inches(1.5), Inches(12), Inches(0.45),
                 font_size=Pt(18), bold=True, color=MID_BLUE)

    insight = (
        "\"Low-level layers capture domain-shared edges and textures.\n"
        " High-level layers encode semantic concepts but are domain-biased.\n"
        " Combining both scales leads to better domain-invariant representations.\""
    )
    add_text_box(s, insight,
                 Inches(0.5), Inches(2.0), Inches(12.3), Inches(1.2),
                 font_size=Pt(15), color=DARK_GRAY)

    # Three boxes
    cols = [Inches(0.4), Inches(4.6), Inches(8.8)]
    titles = ["Early Layers", "Mid Layers", "Late Layers"]
    descs = [
        "Edges, textures,\ncolors — broadly\nshared across domains",
        "Object parts,\nshapes — partially\ndomain-specific",
        "Semantic concepts\n— high domain\nbias, but task-relevant",
    ]
    colors_box = [GREEN, ORANGE, MID_BLUE]
    for col, title, desc, col_c in zip(cols, titles, descs, colors_box):
        add_rect(s, col, Inches(3.4), Inches(4.0), Inches(2.8),
                 fill_color=col_c, line_color=None)
        add_text_box(s, title,
                     col + Inches(0.1), Inches(3.5), Inches(3.8), Inches(0.6),
                     font_size=Pt(18), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(s, desc,
                     col + Inches(0.1), Inches(4.2), Inches(3.8), Inches(1.5),
                     font_size=Pt(14), color=WHITE, align=PP_ALIGN.CENTER)

    add_text_box(s, "M²-CL exploits ALL layers simultaneously via extraction blocks",
                 Inches(0.5), Inches(6.5), Inches(12.3), Inches(0.5),
                 font_size=Pt(15), bold=True, color=DARK_BLUE, align=PP_ALIGN.CENTER)

    add_footer(s, 4)
    return s


def slide_05_method_overview(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Method Overview: M²-CL Architecture",
                   "ResNet backbone with 13 extraction blocks + multi-layer contrastive loss")

    arch = (
        "Input Image (3×224×224)\n"
        "        │\n"
        "   ┌────▼─────────────────────────────────────────────┐\n"
        "   │              ResNet-18 Backbone                   │\n"
        "   │                                                   │\n"
        "   │  Layer1 ──► [EB₁] [EB₂] [EB₃] [EB₄]  u¹..u⁴   │\n"
        "   │  Layer2 ──► [EB₅] [EB₆] [EB₇] [EB₈]  u⁵..u⁸   │\n"
        "   │  Layer3 ──► [EB₉] [EB₁₀] [EB₁₁] [EB₁₂] u⁹..u¹²│\n"
        "   │  Layer4 ──► [EB₁₃]                    u¹³       │\n"
        "   │                   │                              │\n"
        "   │              Global AvgPool                      │\n"
        "   └──────────────────────────────────────────────────┘\n"
        "                       │\n"
        "              ┌────────▼────────┐\n"
        "              │  FC Classifier  │\n"
        "              └────────┬────────┘\n"
        "                       │\n"
        "           ┌───────────▼───────────┐\n"
        "           │  L = L_CE + α·Σ L^(l) │\n"
        "           └───────────────────────┘\n"
    )
    add_text_box(s, arch,
                 Inches(0.4), Inches(1.4), Inches(12.5), Inches(5.6),
                 font_size=Pt(11), color=DARK_BLUE)

    add_footer(s, 5)
    return s


def slide_06_extraction_block(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Extraction Block",
                   "Reduces channels → applies spatial dropout → multi-scale max pooling")

    diagram = (
        "Feature Map  (C × H × W)\n"
        "        │\n"
        "   ┌────▼──────────────┐\n"
        "   │  Conv 1×1 (C→C/r) │  r = 4 (default)\n"
        "   │  BatchNorm + ReLU  │\n"
        "   └────────┬──────────┘\n"
        "            │\n"
        "   ┌────────▼──────────┐\n"
        "   │  Spatial Dropout   │  Drops entire feature maps\n"
        "   └────────┬──────────┘\n"
        "            │\n"
        "    ┌───────┼──────────┐\n"
        "    │       │          │\n"
        " Pool 8×8  Pool 4×4  Pool 2×2    ← Early layers  (3 pipelines)\n"
        "    │       │          │          (Pool 7×7 & 3×3 for later layers)\n"
        "  MLP     MLP        MLP\n"
        "    │       │          │\n"
        "    └───────┴──────────┘\n"
        "            │\n"
        "       Concatenate\n"
        "            │\n"
        "         Linear\n"
        "            │\n"
        "        L2-Norm\n"
        "            │\n"
        "        u^(l)  ∈ ℝ¹²⁸"
    )
    add_text_box(s, diagram,
                 Inches(0.4), Inches(1.4), Inches(12.5), Inches(5.7),
                 font_size=Pt(12), color=DARK_BLUE)

    add_footer(s, 6)
    return s


def slide_07_pipelines(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Concentration Pipelines",
                   "Parallel multi-scale pooling captures different spatial granularities")

    add_text_box(s, "Why parallel (not cascading)?",
                 Inches(0.5), Inches(1.5), Inches(12), Inches(0.45),
                 font_size=Pt(18), bold=True, color=MID_BLUE)

    items = [
        "• Cascading pipelines lose fine-grained spatial information at each stage",
        "• Parallel pipelines independently preserve each spatial resolution",
        "• Ablation result: parallel → 82.83% vs cascading → ~80.0% on PACS",
        "• Early layers (spatial ≥ 8): 3 pipelines at sizes 8×8, 4×4, 2×2",
        "• Later layers (spatial < 8): 2 pipelines at sizes 7×7, 3×3",
    ]
    add_bullet(s, items, Inches(0.5), Inches(2.1), Inches(12.3), Inches(2.8),
               font_size=Pt(16))

    add_text_box(s, "Spatial Dropout Effect",
                 Inches(0.5), Inches(4.4), Inches(12), Inches(0.45),
                 font_size=Pt(18), bold=True, color=MID_BLUE)

    dropout_items = [
        "• Drops entire feature maps (channels), not individual values",
        "• Forces model to learn distributed representations",
        "• With dropout: PACS +1–2.5% over without dropout",
        "• Without dropout: PACS 80.89%, VLCS 73.18%",
    ]
    add_bullet(s, dropout_items, Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0),
               font_size=Pt(16))

    add_footer(s, 7)
    return s


def slide_08_loss(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Multi-Layer Contrastive Loss",
                   "Aligns same-class embeddings across all M intermediate layers")

    add_text_box(s, "Per-layer probability measure for class c at layer l:",
                 Inches(0.5), Inches(1.5), Inches(12.3), Inches(0.45),
                 font_size=Pt(16), color=DARK_GRAY)

    formula1 = (
        "         Σ_{i,j: y_i=y_j=c}  exp( u_i^(l)ᵀ u_j^(l) / τ )\n"
        "p^(l)(c) = ─────────────────────────────────────────────────\n"
        "              Σ_{k,m: k≠m}   exp( u_k^(l)ᵀ u_m^(l) / τ )"
    )
    add_rect(s, Inches(0.4), Inches(2.0), Inches(12.5), Inches(1.3),
             fill_color=LIGHT_GRAY)
    add_text_box(s, formula1,
                 Inches(0.6), Inches(2.05), Inches(12.2), Inches(1.2),
                 font_size=Pt(14), color=DARK_BLUE)

    add_text_box(s, "Layer-wise loss  →  Total loss:",
                 Inches(0.5), Inches(3.5), Inches(12.3), Inches(0.45),
                 font_size=Pt(16), color=DARK_GRAY)

    formula2 = (
        "L^(l) = -Σ_c  log p^(l)(c)\n\n"
        "L  =  L_CE  +  α · Σ_{l=1}^{M}  L^(l)"
    )
    add_rect(s, Inches(0.4), Inches(4.0), Inches(12.5), Inches(1.3),
             fill_color=LIGHT_GRAY)
    add_text_box(s, formula2,
                 Inches(0.6), Inches(4.05), Inches(12.2), Inches(1.2),
                 font_size=Pt(16), bold=True, color=DARK_BLUE)

    params = [
        "• u^(l) : L2-normalized embedding from layer l extraction block",
        "• τ     : temperature (default 1.0, optimal in range 0.1–2.0)",
        "• α     : contrastive loss weight (default 0.01)",
        "• M     : number of layers = 13 (ResNet-18)",
    ]
    add_bullet(s, params, Inches(0.5), Inches(5.4), Inches(12.3), Inches(2.0),
               font_size=Pt(14))

    add_footer(s, 8)
    return s


def slide_09_training(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Training Setup & Hyperparameters",
                   "DomainBed protocol: leave-one-domain-out, 3 random seeds")

    left_title  = "Hyperparameters"
    right_title = "Training Protocol"

    add_text_box(s, left_title,
                 Inches(0.5), Inches(1.5), Inches(5.8), Inches(0.45),
                 font_size=Pt(17), bold=True, color=MID_BLUE)

    rows = [
        ("Backbone",   "ResNet-18 / ResNet-50"),
        ("Pre-training","ImageNet"),
        ("Optimizer",  "SGD (momentum 0.9)"),
        ("LR",         "0.001"),
        ("Epochs",     "30"),
        ("Batch size", "128"),
        ("α",          "0.01"),
        ("τ",          "1.0"),
        ("r (reduction)", "4"),
        ("Dropout p",  "0.5"),
        ("Embed dim",  "128"),
        ("Hardware",   "NVIDIA RTX A5000"),
    ]
    for i, (k, v) in enumerate(rows):
        top = Inches(2.05) + i * Inches(0.38)
        bg = LIGHT_GRAY if i % 2 == 0 else WHITE
        add_rect(s, Inches(0.5), top, Inches(5.8), Inches(0.36), fill_color=bg)
        add_text_box(s, k, Inches(0.6), top, Inches(2.4), Inches(0.36),
                     font_size=Pt(13), bold=True, color=DARK_BLUE)
        add_text_box(s, v, Inches(3.0), top, Inches(3.3), Inches(0.36),
                     font_size=Pt(13), color=DARK_GRAY)

    add_text_box(s, right_title,
                 Inches(7.0), Inches(1.5), Inches(5.8), Inches(0.45),
                 font_size=Pt(17), bold=True, color=MID_BLUE)

    protocol_items = [
        "• Leave-one-domain-out evaluation",
        "• Each domain tested as unseen target",
        "• Average accuracy over 3 random seeds",
        "• Same data splits as DomainBed",
        "• ImageNet pre-trained backbone (frozen BN)",
        "• Cosine annealing LR schedule",
    ]
    add_bullet(s, protocol_items, Inches(7.0), Inches(2.05), Inches(5.8), Inches(2.5),
               font_size=Pt(15))

    add_footer(s, 9)
    return s


def slide_10_datasets(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Benchmark Datasets",
                   "4 standard domain generalization benchmarks")

    datasets = [
        ("PACS",        "4 domains",  "9,991",  "7",  "Photo, Art Painting,\nCartoon, Sketch"),
        ("VLCS",        "4 datasets", "10,729", "5",  "PASCAL, LabelMe,\nCaltech-101, SUN09"),
        ("Office-Home", "4 domains",  "15,588", "65", "Art, Clipart,\nProduct, RealWorld"),
        ("NICO",        "19 classes", "25,000", "19", "Test: 3, 5, or 7\ncontexts held out"),
    ]

    for i, (name, domains, images, classes, notes) in enumerate(datasets):
        col = Inches(0.3) + i * Inches(3.25)
        add_rect(s, col, Inches(1.4), Inches(3.1), Inches(5.6),
                 fill_color=MID_BLUE, line_color=None)
        add_text_box(s, name,
                     col + Inches(0.1), Inches(1.5), Inches(2.9), Inches(0.6),
                     font_size=Pt(22), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_text_box(s, domains,
                     col + Inches(0.1), Inches(2.2), Inches(2.9), Inches(0.45),
                     font_size=Pt(13), color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
        add_text_box(s, f"{images} images",
                     col + Inches(0.1), Inches(2.7), Inches(2.9), Inches(0.45),
                     font_size=Pt(13), color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
        add_text_box(s, f"{classes} classes",
                     col + Inches(0.1), Inches(3.2), Inches(2.9), Inches(0.45),
                     font_size=Pt(13), color=LIGHT_BLUE, align=PP_ALIGN.CENTER)
        add_text_box(s, notes,
                     col + Inches(0.1), Inches(3.8), Inches(2.9), Inches(1.0),
                     font_size=Pt(12), color=WHITE, align=PP_ALIGN.CENTER)

    add_text_box(s, "All evaluated with leave-one-domain-out protocol | Top-1 accuracy",
                 Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.35),
                 font_size=Pt(13), color=DARK_GRAY, align=PP_ALIGN.CENTER)

    add_footer(s, 10)
    return s


def slide_11_results_pacs_vlcs(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Results: PACS & VLCS (ResNet-18)",
                   "M²-CL achieves state-of-the-art on both benchmarks")

    # PACS table
    add_text_box(s, "PACS",
                 Inches(0.4), Inches(1.4), Inches(6.0), Inches(0.4),
                 font_size=Pt(16), bold=True, color=MID_BLUE)

    pacs_data = [
        ("Method",    "Art",    "Cartoon", "Photo",  "Sketch", "Avg"),
        ("ERM",       "77.37",  "75.51",   "96.11",  "69.27",  "84.51"),
        ("CORAL",     "76.23",  "77.01",   "93.58",  "70.35",  "84.29"),
        ("SagNet",    "77.55",  "76.41",   "95.16",  "77.89",  "83.58"),
        ("SAGM",      "79.21",  "77.35",   "94.61",  "79.58",  "82.68"),
        ("M²-CL",     "81.66",  "78.42",   "97.00",  "77.07",  "83.54"),
    ]
    for r, row in enumerate(pacs_data):
        top = Inches(1.85) + r * Inches(0.46)
        bg = DARK_BLUE if r == 0 else (LIGHT_BLUE if r == len(pacs_data) - 1 else
                                        (LIGHT_GRAY if r % 2 == 0 else WHITE))
        add_rect(s, Inches(0.4), top, Inches(6.3), Inches(0.44), fill_color=bg)
        for c, val in enumerate(row):
            fc = WHITE if r == 0 else (DARK_BLUE if r == len(pacs_data) - 1 else DARK_GRAY)
            add_text_box(s, val,
                         Inches(0.4) + c * Inches(1.05), top,
                         Inches(1.05), Inches(0.44),
                         font_size=Pt(12),
                         bold=(r == 0 or r == len(pacs_data) - 1),
                         color=fc, align=PP_ALIGN.CENTER)

    # VLCS table
    add_text_box(s, "VLCS",
                 Inches(7.0), Inches(1.4), Inches(6.0), Inches(0.4),
                 font_size=Pt(16), bold=True, color=MID_BLUE)

    vlcs_data = [
        ("Method",   "PASCAL", "LabelMe", "Caltech", "SUN09", "Avg"),
        ("ERM",      "72.11",  "62.14",   "95.47",   "69.11", "74.71"),
        ("CORAL",    "73.42",  "63.88",   "95.32",   "66.70", "74.83"),
        ("SagNet",   "73.87",  "62.69",   "97.33",   "68.04", "75.48"),
        ("SAGM",     "70.97",  "63.56",   "96.81",   "69.33", "75.17"),
        ("M²-CL",   "73.23",  "64.12",   "98.23",   "75.53", "77.78"),
    ]
    for r, row in enumerate(vlcs_data):
        top = Inches(1.85) + r * Inches(0.46)
        bg = DARK_BLUE if r == 0 else (LIGHT_BLUE if r == len(vlcs_data) - 1 else
                                        (LIGHT_GRAY if r % 2 == 0 else WHITE))
        add_rect(s, Inches(7.0), top, Inches(6.1), Inches(0.44), fill_color=bg)
        for c, val in enumerate(row):
            fc = WHITE if r == 0 else (DARK_BLUE if r == len(vlcs_data) - 1 else DARK_GRAY)
            add_text_box(s, val,
                         Inches(7.0) + c * Inches(1.02), top,
                         Inches(1.02), Inches(0.44),
                         font_size=Pt(12),
                         bold=(r == 0 or r == len(vlcs_data) - 1),
                         color=fc, align=PP_ALIGN.CENTER)

    add_text_box(s, "Bold = M²-CL result    |    +0.86% on PACS avg    |    +2.61% on VLCS avg",
                 Inches(0.5), Inches(5.1), Inches(12.3), Inches(0.4),
                 font_size=Pt(13), color=DARK_GRAY, align=PP_ALIGN.CENTER)

    add_footer(s, 11)
    return s


def slide_12_results_oh_nico(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Results: Office-Home & NICO (ResNet-18)",
                   "M²-CL maintains strong improvements across diverse visual domains")

    # Office-Home table
    add_text_box(s, "Office-Home",
                 Inches(0.4), Inches(1.4), Inches(6.5), Inches(0.4),
                 font_size=Pt(16), bold=True, color=MID_BLUE)

    oh_data = [
        ("Method",   "Art",    "Clipart", "Product", "Real",  "Avg"),
        ("ERM",      "57.45",  "51.74",   "68.11",   "58.27", "58.89"),
        ("CORAL",    "58.09",  "52.92",   "69.17",   "60.19", "60.09"),
        ("SAGM",     "57.92",  "52.65",   "68.95",   "59.12", "59.66"),
        ("M²-CL",   "58.21",  "54.02",   "71.03",   "69.82", "63.27"),
    ]
    for r, row in enumerate(oh_data):
        top = Inches(1.85) + r * Inches(0.46)
        bg = DARK_BLUE if r == 0 else (LIGHT_BLUE if r == len(oh_data) - 1 else
                                        (LIGHT_GRAY if r % 2 == 0 else WHITE))
        add_rect(s, Inches(0.4), top, Inches(6.3), Inches(0.44), fill_color=bg)
        for c, val in enumerate(row):
            fc = WHITE if r == 0 else (DARK_BLUE if r == len(oh_data) - 1 else DARK_GRAY)
            add_text_box(s, val,
                         Inches(0.4) + c * Inches(1.05), top,
                         Inches(1.05), Inches(0.44),
                         font_size=Pt(12),
                         bold=(r == 0 or r == len(oh_data) - 1),
                         color=fc, align=PP_ALIGN.CENTER)

    # NICO table
    add_text_box(s, "NICO (N = held-out contexts)",
                 Inches(7.0), Inches(1.4), Inches(6.0), Inches(0.4),
                 font_size=Pt(16), bold=True, color=MID_BLUE)

    nico_data = [
        ("Method",  "N=3",    "N=5",    "N=7"),
        ("ERM",     "66.77",  "64.11",  "59.10"),
        ("CORAL",   "66.45",  "64.39",  "59.43"),
        ("SagNet",  "66.12",  "64.77",  "60.22"),
        ("M²-CL",  "68.43",  "65.87",  "62.19"),
    ]
    for r, row in enumerate(nico_data):
        top = Inches(1.85) + r * Inches(0.46)
        bg = DARK_BLUE if r == 0 else (LIGHT_BLUE if r == len(nico_data) - 1 else
                                        (LIGHT_GRAY if r % 2 == 0 else WHITE))
        add_rect(s, Inches(7.0), top, Inches(4.5), Inches(0.44), fill_color=bg)
        for c, val in enumerate(row):
            fc = WHITE if r == 0 else (DARK_BLUE if r == len(nico_data) - 1 else DARK_GRAY)
            add_text_box(s, val,
                         Inches(7.0) + c * Inches(1.12), top,
                         Inches(1.12), Inches(0.44),
                         font_size=Pt(12),
                         bold=(r == 0 or r == len(nico_data) - 1),
                         color=fc, align=PP_ALIGN.CENTER)

    add_text_box(s, "+3.61% on Office-Home avg    |    +1.66% / +0.87% / +3.09% on NICO",
                 Inches(0.5), Inches(4.0), Inches(12.3), Inches(0.4),
                 font_size=Pt(13), color=DARK_GRAY, align=PP_ALIGN.CENTER)

    add_footer(s, 12)
    return s


def slide_13_ablation(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Ablation Study",
                   "Each component validated on PACS and VLCS (ResNet-18)")

    ablation_data = [
        ("Configuration",                       "PACS",  "VLCS"),
        ("Full M²-CL (with loss)",              "83.54", "77.78"),
        ("M² only (no contrastive loss)",        "82.83", "76.11"),
        ("Cascading pipelines",                  "80.12", "73.45"),
        ("Without spatial dropout",              "80.89", "73.18"),
        ("r = 2 (more channels)",                "81.73", "75.34"),
        ("r = 6 (fewer channels)",               "81.44", "74.98"),
        ("τ = 0.1 (low temperature)",            "81.67", "76.02"),
        ("τ = 2.0 (high temperature)",           "83.31", "77.61"),
        ("α = 10⁻⁵ (tiny loss weight)",         "82.80", "76.08"),
        ("α = 1.0 (large loss weight)",          "63.54", "57.22"),
    ]

    col_widths = [Inches(7.0), Inches(2.5), Inches(2.5)]
    col_starts = [Inches(0.4), Inches(7.5), Inches(10.1)]

    for r, row in enumerate(ablation_data):
        top = Inches(1.4) + r * Inches(0.5)
        is_header = (r == 0)
        is_best   = (r == 1)
        bg = (DARK_BLUE if is_header else
              LIGHT_BLUE if is_best else
              LIGHT_GRAY if r % 2 == 0 else WHITE)
        add_rect(s, Inches(0.4), top, Inches(12.3), Inches(0.48), fill_color=bg)
        for c, (val, cw, cl) in enumerate(zip(row, col_widths, col_starts)):
            fc = WHITE if is_header else (DARK_BLUE if is_best else DARK_GRAY)
            add_text_box(s, val, cl, top, cw, Inches(0.48),
                         font_size=Pt(12),
                         bold=(is_header or is_best),
                         color=fc,
                         align=PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT)

    add_footer(s, 13)
    return s


def slide_14_conclusion(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "Conclusion",
                   "M²-CL consistently outperforms baselines on all four benchmarks")

    add_text_box(s, "Key Contributions",
                 Inches(0.5), Inches(1.5), Inches(12), Inches(0.45),
                 font_size=Pt(18), bold=True, color=MID_BLUE)

    contributions = [
        "✓  Multi-scale & multi-layer extraction blocks on 13 ResNet layers",
        "✓  Novel parallel concentration pipelines with spatial dropout",
        "✓  Multi-layer contrastive loss that enforces domain-invariant representations",
        "✓  State-of-the-art results: +0.86% PACS, +2.61% VLCS, +3.61% Office-Home",
        "✓  Robust to domain shift even with up to 7 unseen NICO contexts",
    ]
    add_bullet(s, contributions, Inches(0.5), Inches(2.0), Inches(12.3), Inches(2.5),
               font_size=Pt(16), color=DARK_BLUE)

    add_text_box(s, "Limitations & Future Work",
                 Inches(0.5), Inches(4.5), Inches(12), Inches(0.45),
                 font_size=Pt(18), bold=True, color=MID_BLUE)

    future = [
        "→  Extension to ViT / transformer backbones",
        "→  Semi-supervised or few-shot domain settings",
        "→  Video and 3D domain generalization",
    ]
    add_bullet(s, future, Inches(0.5), Inches(5.1), Inches(12.3), Inches(1.5),
               font_size=Pt(16), color=DARK_GRAY)

    add_footer(s, 14)
    return s


def slide_15_references(prs):
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    add_header_bar(s, "References")

    refs = [
        '[1]  A. Ballas & C. Diou. "M2-CL: Multi-Scale and Multi-Layer Contrastive Learning\n'
        '     for Domain Generalization." IEEE Trans. AI, 2024. arXiv:2308.14418',
        "",
        '[2]  I. Gulrajani & D. Lopez-Paz. "In Search of Lost Domain Generalization."\n'
        "     ICLR 2021. (DomainBed framework)",
        "",
        '[3]  K. He et al. "Deep Residual Learning for Image Recognition." CVPR 2016.',
        "",
        '[4]  T. Chen et al. "A Simple Framework for Contrastive Learning of Visual\n'
        '     Representations." ICML 2020.',
        "",
        '[5]  D. Li et al. "Deeper, Broader and Artier Domain Generalization." ICCV 2017.\n'
        "     (PACS dataset)",
        "",
        '[6]  Wang et al. "Generalizing to Unseen Domains: A Survey on Domain\n'
        '     Generalization." IJCAI 2022.',
    ]
    add_bullet(s, refs, Inches(0.5), Inches(1.45), Inches(12.3), Inches(5.5),
               font_size=Pt(13), color=DARK_GRAY)

    add_footer(s, 15)
    return s


def main():
    out_dir = os.path.dirname(os.path.dirname(__file__))
    out_path = os.path.join(out_dir, "m2cl_presentation.pptx")

    prs = new_prs()
    builders = [
        slide_01_title,
        slide_02_outline,
        slide_03_problem,
        slide_04_motivation,
        slide_05_method_overview,
        slide_06_extraction_block,
        slide_07_pipelines,
        slide_08_loss,
        slide_09_training,
        slide_10_datasets,
        slide_11_results_pacs_vlcs,
        slide_12_results_oh_nico,
        slide_13_ablation,
        slide_14_conclusion,
        slide_15_references,
    ]

    for fn in builders:
        fn(prs)

    prs.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
