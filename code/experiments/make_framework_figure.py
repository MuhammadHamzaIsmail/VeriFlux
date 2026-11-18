"""Draws the VeriFlux framework diagram (paper Fig. 1) as a vector PDF + PNG preview.

usage: python code/experiments/make_framework_figure.py
writes paper/figures/framework.pdf and results/figures/framework.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5})

fig, ax = plt.subplots(figsize=(7.16, 4.6))   # IEEE two-column text width
ax.set_xlim(0, 100)
ax.set_ylim(5, 70)
ax.axis("off")

GREY, YEL, BLUE, RED, GREEN, ORANGE = "#ececec", "#fdf8dc", "#e9ebfb", "#fbeceb", "#e3f5e3", "#fdebd5"


def box(cx, cy, w, h, text, fc="white", bold_first=False):
    """Draw a rounded box centred at (cx, cy); returns its edges for arrow routing."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec="black", lw=0.9))
    lines = text.split("\n")
    if bold_first:
        ax.text(cx, cy + h / 2 - 1.9, lines[0], ha="center", va="center", fontweight="bold")
        ax.text(cx, cy - 1.0, "\n".join(lines[1:]), ha="center", va="center", linespacing=1.35)
    else:
        ax.text(cx, cy, text, ha="center", va="center", linespacing=1.35)
    return dict(top=cy + h / 2, bottom=cy - h / 2, x=cx)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=1.1, color="black",
                                shrinkA=0, shrinkB=0))


def line(xs, ys):
    ax.plot(xs, ys, color="black", lw=1.1, solid_capstyle="butt")


# rows (y centres)
data = box(50, 66, 92, 5, "KeepUp dataset: 1,024 posts  ·  172 events  ·  69,507 comments  ·  11,974 evidence documents", GREY)
split = box(50, 58, 92, 5, "Event-grouped split (StratifiedGroupKFold, k = 5, 3 repeats); all fitting on training events only")

ROW_Y, ROW_H = 44, 11
xs = [12, 31, 50, 69, 88]
texts = ["Text embedding\nfrozen XLM-R\ntitle + comments\n$\\mathbf{e}_i \\in \\mathbb{R}^{768}$",
         "Evidence\nBM25 +\nMiniLM-XNLI\n(6 features)",
         "Post–claim\nXNLI +\ncosine\n(5 features)",
         "Stance\nTF-IDF,\ncross-fitted\n(8 features)",
         "User, Eng., Title\nprofile, reactions,\ntitle statistics\n(5 + 9 + 5)"]
fills = [YEL, BLUE, BLUE, BLUE, BLUE]
row = [box(x, ROW_Y, 16.5, ROW_H, t, fc, bold_first=True) for x, t, fc in zip(xs, texts, fills)]

feat = box(59.5, 29, 40, 4.6, "Named features  $\\mathbf{f}_i \\in \\mathbb{R}^{38}$")
contrast = box(50, 19.5, 92, 7,
               "Event-context contrast:  $\\tilde{\\mathbf{v}}_i = \\mathbf{v}_i - \\frac{1}{|S(e_i)|}"
               "\\Sigma_{l \\in S(e_i)}\\, \\mathbf{v}_l$\n(sibling posts of the same claim, labels unused)", RED)
svm = box(28, 9.2, 42, 5.6, "VeriFlux: RBF-SVM on $[\\mathbf{e}, \\tilde{\\mathbf{e}}, \\mathbf{f}, \\tilde{\\mathbf{f}}]$\n"
          "→ fake/real probability (best on unseen events)", GREEN)
lite = box(73, 9.2, 42, 5.6, "VeriFlux-Lite: LightGBM on $[\\mathbf{f}, \\tilde{\\mathbf{f}}, t, \\tilde{t}]$, no transformer\n"
           "→ fake/real + exact TreeSHAP reasons (lowest cost)", GREEN)

# data -> split
arrow(50, data["bottom"], 50, split["top"])
# split -> distribution line -> five boxes
dist_y = 52.5
line([50, 50], [split["bottom"], dist_y])
line([xs[0], xs[-1]], [dist_y, dist_y])
for b in row:
    arrow(b["x"], dist_y, b["x"], b["top"])
# four feature boxes -> collecting line -> named features
bus_y = 34.5
for b in row[1:]:
    line([b["x"], b["x"]], [b["bottom"], bus_y])
line([xs[1], xs[-1]], [bus_y, bus_y])
arrow(feat["x"], bus_y, feat["x"], feat["top"])
# embedding and named features -> contrast
arrow(xs[0], row[0]["bottom"], xs[0], contrast["top"])
arrow(feat["x"], feat["bottom"], feat["x"], contrast["top"])
# contrast -> models
arrow(svm["x"], contrast["bottom"], svm["x"], svm["top"])
arrow(lite["x"], contrast["bottom"], lite["x"], lite["top"])

plt.subplots_adjust(0, 0, 1, 1)
out_pdf = ROOT / "paper" / "figures" / "framework.pdf"
out_png = ROOT / "results" / "figures" / "framework.png"
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote", out_pdf, out_png)
