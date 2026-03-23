import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── configuration — edit these ────────────────────────────────────────────────

ROOT_DIR = "outputs/ablation_man/"  # folder containing s_factor_* subfolders
APPEND_DIR = "orig"  # split to visualize
OUTPUT_PATH = (
    f"outputs/ablation_man_{APPEND_DIR}.png"  # path for the saved output image
)
DPI = 100  # resolution (150–200 for higher quality)
INTERACTIVE = False  # set False to skip the matplotlib window

# ─────────────────────────────────────────────────────────────────────────────


# ── helpers ───────────────────────────────────────────────────────────────────


def parse_s_factor(folder_name: str) -> float:
    """Extract numeric value from folder name like 'sfactor_0.0050'."""
    match = re.search(r"sfactor_([0-9.eE+\-]+)", folder_name)
    if not match:
        raise ValueError(f"Cannot parse sfactor from folder name: {folder_name!r}")
    return float(match.group(1))


def parse_g_scale(file_name: str) -> int:
    """Extract numeric index from file name like 'gscale_01.png'."""
    match = re.search(r"gscale_(\d+)", file_name)
    if not match:
        raise ValueError(f"Cannot parse g_scale from file name: {file_name!r}")
    return int(match.group(1))


def collect_grid(root_dir: str, append_dir: str = ""):
    """
    Returns:
      grid     : dict[(row, col)] -> image file path
      s_labels : sorted list of 's=...' label strings
      g_labels : sorted list of 'g=...' label strings
    """
    subfolders = [
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("sfactor_")
    ]
    if not subfolders:
        raise FileNotFoundError(f"No 'sfactor_*' subfolders found in: {root_dir}")

    entries = []
    for folder in subfolders:
        s_val = parse_s_factor(folder)
        folder_path = os.path.join(root_dir, folder, append_dir)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(".png") and fname.startswith("gscale_"):
                g_idx = parse_g_scale(fname)
                entries.append((s_val, g_idx, os.path.join(folder_path, fname)))

    s_values = sorted(set(e[0] for e in entries))
    g_indices = sorted(set(e[1] for e in entries))

    s_labels = [f"s={v:g}" for v in s_values]
    g_labels = [f"g={i:02d}" for i in g_indices]

    s_map = {v: i for i, v in enumerate(s_values)}
    g_map = {v: i for i, v in enumerate(g_indices)}

    grid = {}
    for s_val, g_idx, path in entries:
        grid[(s_map[s_val], g_map[g_idx])] = path

    return grid, s_labels, g_labels


# ── main visualisation ────────────────────────────────────────────────────────


def build_figure(grid, s_labels, g_labels, img_size_px=512, dpi=100):
    n_rows = len(s_labels)
    n_cols = len(g_labels)

    label_px = 60
    fig_w = (n_cols * img_size_px + label_px) / dpi
    fig_h = (n_rows * img_size_px + label_px) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        left=label_px / (n_cols * img_size_px + label_px),
        bottom=label_px / (n_rows * img_size_px + label_px),
        right=1.0,
        top=1.0,
        hspace=0.04,
        wspace=0.04,
    )

    for row in range(n_rows):
        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            path = grid.get((row, col))
            if path and os.path.exists(path):
                img = Image.open(path).convert("RGB")
                ax.imshow(np.array(img))
            else:
                ax.set_facecolor("#2a2a4a")
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    color="gray",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=7,
                )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if row == 0:
                ax.set_title(
                    g_labels[col], color="white", fontsize=8, pad=4, fontweight="bold"
                )

            if col == 0:
                ax.set_ylabel(
                    s_labels[row],
                    color="white",
                    fontsize=8,
                    rotation=0,
                    labelpad=6,
                    va="center",
                    fontweight="bold",
                )

    fig.suptitle(
        "Image grid: s_factor (rows) × g_scale (columns)",
        color="white",
        fontsize=11,
        y=1.005,
    )
    return fig


# ── entry point ───────────────────────────────────────────────────────────────


def main():
    print(f"Scanning: {ROOT_DIR}")
    grid, s_labels, g_labels = collect_grid(ROOT_DIR, APPEND_DIR)
    print(f"  Found {len(s_labels)} s_factor values × {len(g_labels)} g_scale values")

    fig = build_figure(grid, s_labels, g_labels, dpi=DPI)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved → {OUTPUT_PATH}")

    if INTERACTIVE:
        plt.show()


if __name__ == "__main__":
    main()
