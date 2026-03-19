#!/usr/bin/env python3
"""Shared publication-grade plotting utilities for the TCGA ESCA project."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt

_FONT_FAMILY = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]


def get_nature_palette() -> dict[str, str]:
    """Return the project-wide muted, publication-ready color palette."""
    return {
        "tumor": "#C44E52",
        "normal": "#4C72B0",
        "escc": "#DD8452",
        "eac": "#55A868",
        "smoker": "#8172B3",
        "non_smoker": "#64B5CD",
        "male": "#4C72B0",
        "female": "#C76A9F",
        "early_stage": "#55A868",
        "late_stage": "#C44E52",
        "unknown": "#9A9A9A",
        "train": "#4C72B0",
        "validation": "#55A868",
        "test": "#C44E52",
        "fold_1": "#4C72B0",
        "fold_2": "#55A868",
        "fold_3": "#C44E52",
        "fold_4": "#8172B3",
        "fold_5": "#CCB974",
        "fold_6": "#64B5CD",
        "fold_7": "#8C8C8C",
        "fold_8": "#DA8BC3",
        "fold_9": "#937860",
        "fold_10": "#8C564B",
    }


def configure_publication_plotting() -> dict[str, str]:
    """Set global matplotlib rcParams for publication-grade SVG figures."""
    palette = get_nature_palette()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": _FONT_FAMILY,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "lines.linewidth": 2.0,
            "patch.linewidth": 1.0,
            "svg.fonttype": "none",
            "savefig.format": "svg",
            "savefig.dpi": 300,
            "savefig.transparent": False,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
            "legend.borderaxespad": 0.4,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return palette


def save_publication_figure(
    fig: plt.Figure,
    output_path_svg: str | Path,
    output_path_pdf: str | Path | None = None,
    output_path_png: str | Path | None = None,
) -> None:
    """Save a figure as SVG and optional PDF/PNG exports with tight layout."""
    target_svg = Path(output_path_svg)
    target_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_svg, bbox_inches="tight")
    if output_path_pdf is not None:
        fig.savefig(Path(output_path_pdf), bbox_inches="tight")
    if output_path_png is not None:
        fig.savefig(Path(output_path_png), dpi=600, bbox_inches="tight")


def apply_text_style(ax: plt.Axes) -> None:
    """Force bold serif typography on an existing axis."""
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        text.set_fontweight("bold")
        text.set_fontfamily("serif")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("serif")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")
            text.set_fontfamily("serif")
        title = legend.get_title()
        title.set_fontweight("bold")
        title.set_fontfamily("serif")
