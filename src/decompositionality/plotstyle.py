"""Plot styling shared by the training reports and the diagnostics (torch-free)."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

# Six hue families = slots 1-6 of the validated reference categorical palette, in its
# documented (adjacent-pair validated) order. Each tied baseline uses its
# untied counterpart's hue, lighter and hatched (composite encoding instead of a 9th hue).
_FAMILY = {"ar": "#2a78d6", "enc": "#eb6834", "vit": "#1baf7a", "trm": "#eda100", "res": "#e87ba4", "unet": "#008300"}
MODEL_STYLE = {
    "transformer_ar": dict(color=_FAMILY["ar"], hatch=None, ls="-"),
    "looped_transformer_ar": dict(color=_FAMILY["ar"], hatch="///", ls="--"),
    "transformer_encoder": dict(color=_FAMILY["enc"], hatch=None, ls="-"),
    "looped_transformer_encoder": dict(color=_FAMILY["enc"], hatch="///", ls="--"),
    "vit": dict(color=_FAMILY["vit"], hatch=None, ls="-"),
    "looped_vit": dict(color=_FAMILY["vit"], hatch="///", ls="--"),
    "trm": dict(color=_FAMILY["trm"], hatch=None, ls="-"),
    "rescnn_film": dict(color=_FAMILY["res"], hatch=None, ls="-"),
    "unet_film": dict(color=_FAMILY["unet"], hatch=None, ls="-"),
    "causal_encoder": dict(color="#8e6bd6", hatch=None, ls="-"),
    "looped_causal_encoder": dict(color="#8e6bd6", hatch="///", ls="--"),
    "query_ar": dict(color="#1baf7a", hatch=None, ls="-"),
}
# Fixed categorical colors for digit values 0..9 (digits are also printed in every cell).
DIGIT_COLORS = ["#f2f2f2", "#2a78d6", "#e34948", "#1baf7a", "#eda100", "#8c8c8c",
                "#e87ba4", "#eb6834", "#6ec6e8", "#7a1f2b"]
INVALID_COLOR = "#000000"

def _plt():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.25, "grid.linewidth": 0.6, "font.size": 10, "savefig.dpi": 150,
                         "axes.titleweight": "bold"})
    return plt


def _light(hex_color: str, amount: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    rgb = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def bar_style(model_id: str) -> dict:
    s = MODEL_STYLE[model_id]
    face = _light(s["color"]) if s["hatch"] else s["color"]
    return dict(facecolor=face, edgecolor=s["color"], hatch=s["hatch"], linewidth=0.8)


def save_fig(fig, out: Path, name: str, data=None) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    if data is not None:
        data.to_csv(out / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def draw_grid(ax, grid, title=None, errors=None):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    g = np.asarray(grid).reshape(6, 6).astype(int)
    shown = np.where(g < 0, 10, g)
    ax.imshow(shown, cmap=ListedColormap(DIGIT_COLORS + [INVALID_COLOR]), vmin=-0.5, vmax=10.5)
    for r in range(6):
        for c in range(6):
            v = g[r, c]
            txt = "?" if v < 0 else str(v)
            light_bg = v in (0, 4, 5, 6, 8) and v >= 0
            ax.text(c, r, txt, ha="center", va="center", fontsize=7, color="#0b0b0b" if light_bg else "white")
            if errors is not None and errors[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="#ff0000", lw=2.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, fontweight="normal")


