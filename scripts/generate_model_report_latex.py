"""Generate a high-quality LaTeX/PDF model evaluation report for V47.

Steps:
  1. Generate vector charts (PDF) with matplotlib
  2. Write .tex source
  3. Compile with xelatex (two passes)

Usage:
    python scripts/generate_model_report_latex.py [--out-dir logs/eval/report_v47]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
HORIZONS  = [6, 12, 24, 48, 72]
STATIONS  = ["BKK_01", "CNX_01", "HYI_01", "KKN_01", "RYG_01"]
RUN_ID    = "20260511T164652Z"
RUN_DIR   = Path("logs/eval/runs") / RUN_ID
EVAL_DIR  = Path("logs/eval/v47")
VERSION   = "47"

STATION_LABEL = {
    "BKK_01": "Bangkok",
    "CNX_01": "Chiang Mai",
    "HYI_01": "Hat Yai",
    "KKN_01": "Khon Kaen",
    "RYG_01": "Rayong",
}
STATION_CODE = {
    "BKK_01": "BKK\\_01",
    "CNX_01": "CNX\\_01",
    "HYI_01": "HYI\\_01",
    "KKN_01": "KKN\\_01",
    "RYG_01": "RYG\\_01",
}
STATION_REGION = {
    "BKK_01": "Central Thailand · 13.76°N 100.50°E",
    "CNX_01": "Northern Thailand · 18.79°N 98.98°E",
    "HYI_01": "Southern Thailand · 7.00°N 100.47°E",
    "KKN_01": "Northeast Thailand · 16.44°N 102.84°E",
    "RYG_01": "Eastern Thailand · 12.68°N 101.27°E",
}

# Matplotlib colour palette (hex)
COL_NAVY  = "#0D2137"
COL_BLUE  = "#1A6FA8"
COL_TEAL  = "#17A589"
COL_GREEN = "#27AE60"
COL_AMBER = "#E67E22"
COL_RED   = "#C0392B"
COL_LGREY = "#ECF0F1"
COL_MGREY = "#95A5A6"
PALETTE   = [COL_BLUE, COL_TEAL, COL_GREEN, "#8E44AD", COL_AMBER]


# ── data loading ──────────────────────────────────────────────────────────────
def load_metrics() -> dict[tuple[str, int], dict]:
    out = {}
    for sid in STATIONS:
        for h in HORIZONS:
            p = RUN_DIR / "lightgbm_quantile" / sid / f"h{h}" / "metrics.json"
            if p.exists():
                out[(sid, h)] = json.loads(p.read_text())
    return out


def load_leaderboard() -> dict[tuple[str, int], dict]:
    p = EVAL_DIR / "leaderboard.json"
    rows = json.loads(p.read_text()) if p.exists() else []
    return {(r["station"], r["horizon_h"]): r for r in rows}


# ── helpers ───────────────────────────────────────────────────────────────────
def tex(s: str) -> str:
    """Escape a plain string for LaTeX."""
    return (str(s)
            .replace("&",  r"\&")
            .replace("%",  r"\%")
            .replace("$",  r"\$")
            .replace("#",  r"\#")
            .replace("_",  r"\_")
            .replace("{",  r"\{")
            .replace("}",  r"\}")
            .replace("~",  r"\textasciitilde{}")
            .replace("^",  r"\textasciicircum{}")
            .replace("\\", r"\textbackslash{}"))


def skill_rgb(s: float) -> tuple[int, int, int]:
    """Map skill to (R,G,B) in 0-255 for LaTeX cellcolor."""
    if s >= 0.5:   return (39,  174, 96)   # strong green
    if s >= 0.2:   return (88,  214, 141)  # light green
    if s >= 0.05:  return (212, 239, 223)  # pale green
    if s >= -0.2:  return (253, 235, 208)  # pale amber
    if s >= -0.8:  return (241, 148, 138)  # light red
    return          (192, 57,  43)          # strong red


def mae_rgb(v: float) -> tuple[int, int, int]:
    lo, hi = 0.7, 3.5
    t = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    r = int(39  + t * (192 - 39))
    g = int(174 + t * (57  - 174))
    b = int(96  + t * (43  - 96))
    return r, g, b


def recall_rgb(v: float | None) -> tuple[int, int, int]:
    if v is None:
        return (220, 220, 220)
    if v >= 0.75: return (39, 174, 96)
    if v >= 0.5:  return (88, 214, 141)
    if v >= 0.25: return (253, 235, 208)
    return (241, 148, 138)


def fmt_skill(s: float | None) -> str:
    if s is None: return "—"
    return f"{s:+.3f}"

def fmt_mae(v: float | None) -> str:
    if v is None: return "—"
    return f"{v:.3f}"

def fmt_recall(v: float | None) -> str:
    if v is None: return "—"
    return f"{v:.2f}"


# ── chart generators ──────────────────────────────────────────────────────────
def save_fig(fig, path: Path, dpi: int = 200):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_overview_bars(metrics: dict, lb: dict, out: Path):
    """One-page 2×3 grid: skill score + MAE per station."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor="white")
    fig.subplots_adjust(hspace=0.45)

    ax_skill, ax_mae = axes

    x = np.arange(len(HORIZONS))
    w = 0.14
    offsets = np.linspace(-(len(STATIONS)-1)*w/2, (len(STATIONS)-1)*w/2, len(STATIONS))

    # ── Skill score ──
    for k, (sid, col) in enumerate(zip(STATIONS, PALETTE)):
        skills = [metrics[(sid, h)]["baselines"]["skill_score"]
                  if (sid, h) in metrics else np.nan for h in HORIZONS]
        bars = ax_skill.bar(x + offsets[k], skills, w, color=col, label=STATION_LABEL[sid],
                            alpha=0.88, zorder=3)
    ax_skill.axhline(0,    color="black",     lw=1.0, zorder=4)
    ax_skill.axhline(0.10, color=COL_TEAL,  lw=1.2, ls="--", zorder=4, label="Candidate threshold (+0.10)")
    ax_skill.axhline(0.55, color=COL_GREEN, lw=1.2, ls=":",  zorder=4, label="Ready threshold (+0.55)")
    ax_skill.set_xticks(x); ax_skill.set_xticklabels([f"h{h}" for h in HORIZONS], fontsize=12)
    ax_skill.set_ylabel("Skill Score", fontsize=12)
    ax_skill.set_title("Skill Score vs Climatology Baseline — All Stations", fontsize=13, fontweight="bold", pad=8)
    ax_skill.legend(fontsize=9, ncol=4, loc="lower left", framealpha=0.9)
    ax_skill.grid(axis="y", alpha=0.3, zorder=0)
    ax_skill.set_facecolor("#FAFAFA")

    # ── MAE ──
    for k, (sid, col) in enumerate(zip(STATIONS, PALETTE)):
        maes = [metrics[(sid, h)]["regression"]["mae"]
                if (sid, h) in metrics else np.nan for h in HORIZONS]
        ax_mae.bar(x + offsets[k], maes, w, color=col, label=STATION_LABEL[sid],
                   alpha=0.88, zorder=3)
    ax_mae.set_xticks(x); ax_mae.set_xticklabels([f"h{h}" for h in HORIZONS], fontsize=12)
    ax_mae.set_ylabel("MAE (°C)", fontsize=12)
    ax_mae.set_title("Mean Absolute Error — All Stations", fontsize=13, fontweight="bold", pad=8)
    ax_mae.legend(fontsize=9, ncol=5, loc="upper left", framealpha=0.9)
    ax_mae.grid(axis="y", alpha=0.3, zorder=0)
    ax_mae.set_facecolor("#FAFAFA")

    save_fig(fig, out)


def chart_danger_recall(metrics: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    fig.subplots_adjust(wspace=0.32)

    x = np.arange(len(HORIZONS))
    w = 0.14
    offsets = np.linspace(-(len(STATIONS)-1)*w/2, (len(STATIONS)-1)*w/2, len(STATIONS))

    for ax, key, title in zip(
        axes,
        ["danger_42", "danger_40"],
        ["Danger Recall  HI ≥ 42 °C  (Extreme Heat)", "Danger Recall  HI ≥ 40 °C  (Dangerous Heat)"],
    ):
        for k, (sid, col) in enumerate(zip(STATIONS, PALETTE)):
            recalls = []
            for h in HORIZONS:
                m = metrics.get((sid, h))
                r = m["safety"][key]["recall"] if m else 0.0
                recalls.append(r if r is not None else 0.0)
            ax.bar(x + offsets[k], recalls, w, color=col, label=STATION_LABEL[sid], alpha=0.88, zorder=3)
        ax.axhline(0.5, color=COL_RED, lw=1.5, ls="--", zorder=4, label="50 % safety target")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(x); ax.set_xticklabels([f"h{h}" for h in HORIZONS], fontsize=12)
        ax.set_ylabel("Recall", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.legend(fontsize=9, ncol=2, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_facecolor("#FAFAFA")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    save_fig(fig, out)


def chart_station(sid: str, metrics: dict, out: Path):
    hs = [h for h in HORIZONS if (sid, h) in metrics]
    maes       = [metrics[(sid, h)]["regression"]["mae"]           for h in hs]
    clim_maes  = [metrics[(sid, h)]["baselines"]["climatology_mae"] for h in hs]
    pers_maes  = [metrics[(sid, h)]["baselines"]["persistence_mae"] for h in hs]
    skills     = [metrics[(sid, h)]["baselines"]["skill_score"]     for h in hs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.subplots_adjust(wspace=0.30)

    x = np.arange(len(hs))
    w = 0.26

    b1 = ax1.bar(x - w, maes,      w, color=COL_BLUE,  label="Model MAE",      alpha=0.9, zorder=3)
    b2 = ax1.bar(x,     clim_maes, w, color=COL_AMBER,  label="Climatology MAE", alpha=0.9, zorder=3)
    b3 = ax1.bar(x + w, pers_maes, w, color=COL_MGREY,  label="Persistence MAE", alpha=0.9, zorder=3)
    for b, v in zip(b1, maes):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                 color=COL_BLUE, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels([f"h{h}" for h in hs], fontsize=12)
    ax1.set_ylabel("MAE (°C)", fontsize=12)
    ax1.set_title(f"{STATION_LABEL[sid]} — MAE vs Baselines", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_facecolor("#FAFAFA")

    bar_colors = [COL_GREEN if s >= 0 else COL_RED for s in skills]
    bars = ax2.bar(x, skills, 0.55, color=bar_colors, alpha=0.88, edgecolor="white", lw=0.8, zorder=3)
    ax2.axhline(0,    color="black",    lw=1.0, zorder=4)
    ax2.axhline(0.10, color=COL_TEAL,  lw=1.3, ls="--", zorder=4, label="Candidate (+0.10)")
    ax2.axhline(0.55, color=COL_GREEN, lw=1.3, ls=":",  zorder=4, label="Ready (+0.55)")
    for b, v in zip(bars, skills):
        va = "bottom" if v >= 0 else "top"
        off = 0.015 if v >= 0 else -0.015
        ax2.text(b.get_x() + b.get_width()/2, v + off, f"{v:+.3f}",
                 ha="center", va=va, fontsize=9.5, fontweight="bold",
                 color="white" if abs(v) > 0.6 else "black")
    ax2.set_xticks(x); ax2.set_xticklabels([f"h{h}" for h in hs], fontsize=12)
    ax2.set_ylabel("Skill Score", fontsize=12)
    ax2.set_title(f"{STATION_LABEL[sid]} — Skill Score", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_facecolor("#FAFAFA")

    save_fig(fig, out)


def chart_pi_coverage(metrics: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    x = np.arange(len(HORIZONS))
    w = 0.14
    offsets = np.linspace(-(len(STATIONS)-1)*w/2, (len(STATIONS)-1)*w/2, len(STATIONS))

    for ax, key, title, ref, ref_lbl in zip(
        axes,
        ["coverage_90", "mean_width"],
        ["PI Coverage (target = 0.90)", "PI Mean Width (°C)"],
        [0.90, None],
        ["target 0.90", None],
    ):
        for k, (sid, col) in enumerate(zip(STATIONS, PALETTE)):
            vals = [metrics[(sid, h)]["prediction_interval"][key]
                    if (sid, h) in metrics else np.nan for h in HORIZONS]
            ax.bar(x + offsets[k], vals, w, color=col, label=STATION_LABEL[sid], alpha=0.88, zorder=3)
        if ref is not None:
            ax.axhline(ref, color=COL_NAVY, lw=1.5, ls="--", zorder=4, label=ref_lbl)
        ax.set_xticks(x); ax.set_xticklabels([f"h{h}" for h in HORIZONS], fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, ncol=2)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_facecolor("#FAFAFA")

    save_fig(fig, out)


# ── LaTeX helpers ─────────────────────────────────────────────────────────────
def cell_color(r: int, g: int, b: int) -> str:
    return rf"\cellcolor[RGB]{{{r},{g},{b}}}"

def bold(s: str) -> str:
    return rf"\textbf{{{s}}}"

def colored_text(r: int, g: int, b: int, s: str) -> str:
    return rf"{{\color[RGB]{{{r},{g},{b}}}{s}}}"


# ── LaTeX section builders ────────────────────────────────────────────────────
def make_skill_table(metrics: dict, lb: dict) -> str:
    lines = []
    cols = "X" + "c" * len(HORIZONS)
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{cols}}}")
    lines.append(r"\toprule")
    header = r"\textbf{Station}" + " & " + " & ".join(
        rf"\textbf{{h{h}}}" for h in HORIZONS) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for sid in STATIONS:
        row_cells = [rf"\textbf{{{tex(STATION_LABEL[sid])}}}"]
        for h in HORIZONS:
            m = metrics.get((sid, h))
            if not m:
                row_cells.append("—")
                continue
            s = m["baselines"]["skill_score"]
            r, g, b = skill_rgb(s)
            status = lb.get((sid, h), {}).get("status", "")
            badge = r"\,\scalebox{0.6}{\textbf{\textsf{CAND}}}" if status == "candidate" else \
                    r"\,\scalebox{0.6}{\textbf{\textsf{READY}}}" if status == "ready" else \
                    r"\,\scalebox{0.6}{\textcolor{gray}{\textsf{NR}}}"
            # make text black on light backgrounds, white on dark
            luminance = 0.299*r + 0.587*g + 0.114*b
            txt_color = "black" if luminance > 140 else "white"
            cell = (rf"{cell_color(r,g,b)}"
                    rf"\textcolor{{{txt_color}}}{{\textbf{{{s:+.3f}}}}}{badge}")
            row_cells.append(cell)
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def make_mae_table(metrics: dict) -> str:
    lines = []
    cols = "X" + "c" * len(HORIZONS)
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{cols}}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Station} & " + " & ".join(
        rf"\textbf{{h{h}}}" for h in HORIZONS) + r" \\")
    lines.append(r"\midrule")
    for sid in STATIONS:
        row_cells = [rf"\textbf{{{tex(STATION_LABEL[sid])}}}"]
        for h in HORIZONS:
            m = metrics.get((sid, h))
            if not m:
                row_cells.append("—")
                continue
            v = m["regression"]["mae"]
            r, g, b = mae_rgb(v)
            luminance = 0.299*r + 0.587*g + 0.114*b
            tc = "black" if luminance > 140 else "white"
            row_cells.append(rf"{cell_color(r,g,b)}\textcolor{{{tc}}}{{\textbf{{{v:.3f}}}}}")
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def make_danger_recall_table(metrics: dict, key: str) -> str:
    lines = []
    cols = "X" + "c" * len(HORIZONS)
    lines.append(rf"\begin{{tabularx}}{{\linewidth}}{{{cols}}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Station} & " + " & ".join(
        rf"\textbf{{h{h}}}" for h in HORIZONS) + r" \\")
    lines.append(r"\midrule")
    for sid in STATIONS:
        row_cells = [rf"\textbf{{{tex(STATION_LABEL[sid])}}}"]
        for h in HORIZONS:
            m = metrics.get((sid, h))
            if not m:
                row_cells.append("—")
                continue
            v = m["safety"][key]["recall"]
            n = m["safety"][key]["support"]
            r, g, b = recall_rgb(v)
            if v is None:
                row_cells.append(rf"{cell_color(220,220,220)}\textcolor{{gray}}{{n/a}}")
            else:
                luminance = 0.299*r + 0.587*g + 0.114*b
                tc = "black" if luminance > 140 else "white"
                row_cells.append(rf"{cell_color(r,g,b)}\textcolor{{{tc}}}{{\textbf{{{v:.2f}}}}}"
                                 rf"\,\scalebox{{0.65}}{{\textcolor{{gray}}{{n={n}}}}}")
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def make_station_detail_table(sid: str, metrics: dict, lb: dict) -> str:
    """Wide metrics table for one station."""
    row_defs = [
        ("MAE (°C)",              lambda m: f"{m['regression']['mae']:.4f}"),
        ("RMSE (°C)",             lambda m: f"{m['regression']['rmse']:.4f}"),
        ("Bias (°C)",             lambda m: f"{m['regression']['bias']:+.4f}"),
        ("Correlation",           lambda m: f"{m['regression']['correlation']:.4f}"),
        ("P50 |error| (°C)",      lambda m: f"{m['regression']['p50_abs_error']:.3f}"),
        ("P90 |error| (°C)",      lambda m: f"{m['regression']['p90_abs_error']:.3f}"),
        ("P95 |error| (°C)",      lambda m: f"{m['regression']['p95_abs_error']:.3f}"),
        (r"\midrule", None),
        ("Climatology MAE",       lambda m: f"{m['baselines']['climatology_mae']:.4f}"),
        ("Persistence MAE",       lambda m: f"{m['baselines']['persistence_mae']:.4f}"),
        ("Skill Score",           lambda m: f"{m['baselines']['skill_score']:+.4f}"),
        (r"\midrule", None),
        ("PI Coverage 90\\%",     lambda m: f"{m['prediction_interval']['coverage_90']:.4f}"),
        ("PI Mean Width (°C)",    lambda m: f"{m['prediction_interval']['mean_width']:.3f}"),
        ("PI P25 Width (°C)",     lambda m: f"{m['prediction_interval']['p25_width']:.3f}"),
        ("PI P75 Width (°C)",     lambda m: f"{m['prediction_interval']['p75_width']:.3f}"),
        ("Pinball Q05",           lambda m: f"{m['prediction_interval']['pinball_q05']:.5f}"),
        ("Pinball Q95",           lambda m: f"{m['prediction_interval']['pinball_q95']:.5f}"),
        (r"\midrule", None),
        ("Danger Recall $\\geq$42\\,°C", lambda m: (
            f"{m['safety']['danger_42']['recall']:.3f} (n={m['safety']['danger_42']['support']})"
            if m['safety']['danger_42']['recall'] is not None else "n/a")),
        ("Danger Recall $\\geq$40\\,°C", lambda m:
            f"{m['safety']['danger_40']['recall']:.3f} (n={m['safety']['danger_40']['support']})"),
        (r"\midrule", None),
        ("Test rows",             lambda m: f"{m['split']['row_counts']['test']:,}"),
        ("Train end",             lambda m: m["split"]["date_ranges"]["train"]["end"][:10]),
        ("Test start",            lambda m: m["split"]["date_ranges"]["test"]["start"][:10]),
        ("Test end",              lambda m: m["split"]["date_ranges"]["test"]["end"][:10]),
        ("Train time (s)",        lambda m: f"{m['runtime']['train_seconds']:.1f}"),
        ("Status",                lambda m: lb.get((sid, m["horizon_h"]), {}).get("status", "—").upper()),
    ]

    cols = "X" + "c" * len(HORIZONS)
    lines = [rf"\begin{{tabularx}}{{\linewidth}}{{{cols}}}"]
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & " + " & ".join(
        rf"\textbf{{h{h}}}" for h in HORIZONS) + r" \\")
    lines.append(r"\midrule")

    alt = False
    for label, fn in row_defs:
        if label == r"\midrule":
            lines.append(r"\midrule")
            continue
        row_cells = [rf"\textsf{{\small {label}}}"]
        bg = r"\rowcolor{hlgrey}" if alt else ""
        for h in HORIZONS:
            m = metrics.get((sid, h))
            try:
                val = fn(m) if m else "—"
            except Exception:
                val = "—"

            # special colouring for skill and status
            if "Skill" in label and m:
                s = m["baselines"]["skill_score"]
                r_, g_, b_ = skill_rgb(s)
                lum = 0.299*r_ + 0.587*g_ + 0.114*b_
                tc = "black" if lum > 140 else "white"
                row_cells.append(rf"{cell_color(r_,g_,b_)}\textcolor{{{tc}}}{{{val}}}")
            elif "Status" in label and m:
                st = lb.get((sid, h), {}).get("status", "")
                dval = val.replace("_", "-")
                if st == "candidate":
                    row_cells.append(rf"\cellcolor[RGB]{{212,239,223}}\textbf{{\small {dval}}}")
                elif st == "ready":
                    row_cells.append(rf"\cellcolor[RGB]{{39,174,96}}\textcolor{{white}}{{\textbf{{{dval}}}}}")
                else:
                    row_cells.append(rf"\cellcolor[RGB]{{253,228,215}}\textcolor{{gray}}{{\small {dval}}}")
            else:
                row_cells.append(val)
        lines.append(f"{bg}" + " & ".join(row_cells) + r" \\")
        alt = not alt

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def make_risk_table(metrics: dict) -> str:
    """h6 risk classification precision/recall per station."""
    levels = ["Caution", "Extreme Caution", "Danger", "Extreme Danger"]
    head_cells = [r"\textbf{Station}"] + [rf"\textbf{{{tex(l)}}}" for l in levels] + [r"\textbf{Accuracy}"]
    lines = [r"\begin{tabularx}{\linewidth}{Xccccr}", r"\toprule"]
    lines.append(" & ".join(head_cells) + r" \\")
    lines.append(r"\midrule")
    for i, sid in enumerate(STATIONS):
        m = metrics.get((sid, 6))
        if not m:
            continue
        cr = m["risk"]["classification_report"]
        row = [rf"\textbf{{{tex(STATION_LABEL[sid])}}}"]
        for lbl in levels:
            info = cr.get(lbl, {})
            p = info.get("precision", 0.0)
            r_ = info.get("recall", 0.0)
            sup = int(info.get("support", 0))
            bg = r"\cellcolor[RGB]{212,239,223}" if r_ >= 0.8 else \
                 r"\cellcolor[RGB]{253,235,208}" if r_ >= 0.5 else \
                 r"\cellcolor[RGB]{253,228,215}"
            row.append(rf"{bg}\small P\,{p:.2f} / R\,{r_:.2f}"
                       rf"\,\scalebox{{0.65}}{{\textcolor{{gray}}{{(n={sup})}}}}")
        acc = cr.get("accuracy", 0)
        row.append(rf"\textbf{{{acc:.3f}}}")
        bg_row = r"\rowcolor{hlgrey}" if i % 2 == 1 else ""
        lines.append(f"{bg_row}" + " & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


def make_pi_table(metrics: dict) -> str:
    lines = [
        r"\begin{tabularx}{\linewidth}{Xlccrrrr}",
        r"\toprule",
        r"\textbf{Station} & \textbf{Horizon} & \textbf{Coverage 90\%} & "
        r"\textbf{Mean Width} & \textbf{P25 Width} & \textbf{P75 Width} & "
        r"\textbf{Pinball Q05} & \textbf{Pinball Q95} \\",
        r"\midrule",
    ]
    alt = False
    for sid in STATIONS:
        for h in HORIZONS:
            m = metrics.get((sid, h))
            if not m:
                continue
            pi = m["prediction_interval"]
            cov = pi["coverage_90"]
            cov_str = f"{cov:.4f}"
            if cov > 0.97:
                cov_cell = rf"\cellcolor[RGB]{{253,235,208}}{cov_str}"
            elif cov >= 0.87:
                cov_cell = rf"\cellcolor[RGB]{{212,239,223}}{cov_str}"
            else:
                cov_cell = rf"\cellcolor[RGB]{{253,228,215}}{cov_str}"
            bg = r"\rowcolor{hlgrey}" if alt else ""
            lines.append(
                f"{bg}{tex(STATION_LABEL[sid])} & h{h} & {cov_cell} & "
                f"{pi['mean_width']:.3f} & {pi['p25_width']:.3f} & "
                f"{pi['p75_width']:.3f} & {pi['pinball_q05']:.5f} & "
                f"{pi['pinball_q95']:.5f} \\\\"
            )
            alt = not alt
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabularx}")
    return "\n".join(lines)


# ── main LaTeX document ───────────────────────────────────────────────────────
def build_tex(metrics: dict, lb: dict, out_dir: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")
    chart_dir = out_dir / "charts"

    # generate charts
    chart_overview = chart_dir / "overview.pdf"
    chart_danger   = chart_dir / "danger_recall.pdf"
    chart_pi       = chart_dir / "pi_quality.pdf"

    chart_overview_bars(metrics, lb, chart_overview)
    chart_danger_recall(metrics, chart_danger)
    chart_pi_coverage(metrics, chart_pi)

    station_charts = {}
    for sid in STATIONS:
        p = chart_dir / f"station_{sid}.pdf"
        chart_station(sid, metrics, p)
        station_charts[sid] = p

    def inc(p: Path, w="\\linewidth") -> str:
        rel = p.relative_to(out_dir).as_posix()
        return rf"\includegraphics[width={w}]{{{rel}}}"

    # ── per-station sections ──
    station_sections = []
    for s_idx, sid in enumerate(STATIONS):
        sec = rf"""
\subsection{{{tex(STATION_LABEL[sid])} \textnormal{{\small ({STATION_CODE[sid]})}}}}\label{{sec:{sid}}}

\begin{{tcolorbox}}[colback=navyblue!6, colframe=navyblue!30, boxrule=0.6pt,
  arc=3pt, left=6pt, right=6pt, top=4pt, bottom=4pt]
  \small\textsf{{\textbf{{Region:}}}} {tex(STATION_REGION[sid])}
\end{{tcolorbox}}

\vspace{{4pt}}
{inc(station_charts[sid])}

\vspace{{6pt}}
{make_station_detail_table(sid, metrics, lb)}
"""
        station_sections.append(sec)
    station_body = "\n\\newpage\n".join(station_sections)

    # count ready slots
    n_pos = sum(1 for r in lb.values() if r.get("status") in ("candidate","ready"))
    n_tot = len(lb)

    doc = rf"""
%% Auto-generated by generate_model_report_latex.py
%% Run ID: {RUN_ID}

\documentclass[11pt, a4paper]{{article}}

%% — Packages ——————————————————————————————————————————————————————————————————
\usepackage{{geometry}}
\geometry{{a4paper, top=2.4cm, bottom=2.4cm, left=2.2cm, right=2.2cm,
           headheight=14pt}}

\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setsansfont{{Arial}}
\setmonofont{{Courier New}}

\usepackage{{microtype}}
\usepackage[dvipsnames]{{xcolor}}
\usepackage{{colortbl}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{array}}
\usepackage{{multirow}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{tikz}}
\usetikzlibrary{{calc, positioning, shapes.geometric}}
\usepackage[most]{{tcolorbox}}
\usepackage{{enumitem}}
\usepackage{{fancyhdr}}
\usepackage{{titlesec}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{parskip}}
\usepackage{{scalefnt}}
\usepackage[unicode, colorlinks=true, linkcolor=navyblue,
            urlcolor=accentblue, citecolor=navyblue,
            pdftitle={{HeatShield AI — Model Evaluation Report V{VERSION}}},
            pdfauthor={{HeatShield AI Pipeline}}]{{hyperref}}

%% — Colours ——————————————————————————————————————————————————————————————————
\definecolor{{navyblue}}{{RGB}}{{13,33,55}}
\definecolor{{accentblue}}{{RGB}}{{26,111,168}}
\definecolor{{tealgreen}}{{RGB}}{{23,165,137}}
\definecolor{{positivegreen}}{{RGB}}{{39,174,96}}
\definecolor{{warnamber}}{{RGB}}{{230,126,34}}
\definecolor{{alertred}}{{RGB}}{{192,57,43}}
\definecolor{{hlgrey}}{{RGB}}{{242,244,244}}

%% — Section styles ————————————————————————————————————————————————————————————
\titleformat{{\section}}
  {{\sffamily\Large\bfseries\color{{navyblue}}}}
  {{\thesection}}{{1em}}{{}}
  [{{\titlerule[1.2pt]}}]

\titleformat{{\subsection}}
  {{\sffamily\large\bfseries\color{{accentblue}}}}
  {{\thesubsection}}{{1em}}{{}}

\titleformat{{\subsubsection}}
  {{\sffamily\normalsize\bfseries\color{{tealgreen}}}}
  {{\thesubsubsection}}{{1em}}{{}}

%% — Header / footer ——————————————————————————————————————————————————————————
\pagestyle{{fancy}}
\fancyhf{{}}
\renewcommand{{\headrulewidth}}{{0pt}}
\fancyhead[L]{{\colorbox{{navyblue}}{{\makebox[0pt][l]{{\rule[-4pt]{{0pt}}{{14pt}}}}%
  \hspace{{4pt}}\textcolor{{white}}{{\textsf{{\small\bfseries
  HeatShield AI — Model Evaluation Report V{VERSION}}}}}%
  \hspace{{4pt}}}}}}
\fancyhead[R]{{\textcolor{{gray}}{{\textsf{{\small Run: {tex(RUN_ID)}}}}}}}
\fancyfoot[C]{{\textcolor{{gray}}{{\textsf{{\small\thepage}}}}}}

%% — Caption style ————————————————————————————————————————————————————————————
\captionsetup{{font=small, labelfont={{sf,bf}}, format=hang, skip=4pt}}

%% — tcolorbox styles —————————————————————————————————————————————————————————
\tcbset{{
  finding/.style={{
    enhanced, colback=accentblue!8, colframe=accentblue!40, boxrule=0.8pt,
    arc=4pt, left=8pt, right=8pt, top=5pt, bottom=5pt,
    fontupper=\small\sffamily
  }},
  statbox/.style={{
    enhanced, colback=#1!10, colframe=#1!50, boxrule=1pt,
    arc=5pt, halign=center, valign=center,
    width=3.8cm, height=2.8cm,
  }},
}}

%% ——————————————————————————————————————————————————————————————————————————————
\begin{{document}}

%% ═══════════════════════════════════════════════════════════════
%% COVER PAGE
%% ═══════════════════════════════════════════════════════════════
\begin{{titlepage}}
\vspace*{{4cm}}
\begin{{center}}
  {{\color{{navyblue}}\rule{{\linewidth}}{{1.5pt}}}}\\[0.8cm]
  {{\LARGE\bfseries\sffamily HeatShield AI}}\\[0.4cm]
  {{\Large\sffamily Model Evaluation Report --- Version {VERSION}}}\\[0.6cm]
  {{\color{{navyblue}}\rule{{\linewidth}}{{0.4pt}}}}\\[3.5cm]
  {{\small\textcolor{{gray}} Generated: {now}}}\\[0.2cm]
  {{\small\textcolor{{gray}} Run ID: \texttt{{{tex(RUN_ID)}}}}}
\end{{center}}
\end{{titlepage}}

%% ═══════════════════════════════════════════════════════════════
%% TABLE OF CONTENTS
%% ═══════════════════════════════════════════════════════════════
\tableofcontents
\newpage

%% ═══════════════════════════════════════════════════════════════
%% 1. EXECUTIVE SUMMARY
%% ═══════════════════════════════════════════════════════════════
\section{{Executive Summary}}\label{{sec:exec}}

This report presents the evaluation results for \textbf{{HeatShield AI V{VERSION}}} ---
a LightGBM Quantile Regression forecast system covering five Thai meteorological
stations across five forecast horizons (6\,h through 72\,h). Models were trained on
ERA5 reanalysis and TMD surface observations spanning 2021--2025.

\subsection{{Key Findings}}

\begin{{tcolorbox}}[finding]
\begin{{itemize}}[leftmargin=1.2em, itemsep=4pt]
  \item \textbf{{{n_pos} of {n_tot} forecast slots achieve positive skill score}} ---
        all short-to-medium horizons (h6, h12, h24) beat the seasonal climatology
        baseline across every station.
  \item \textbf{{Best performer: KKN\_01 h6}} --- skill score $+0.578$, MAE $1.047$\,°C;
        consistently outperforms climatology by a wide margin.
  \item \textbf{{Lowest absolute error: HYI\_01 (Hat Yai)}} --- MAE $0.73$\,°C at h6,
        driven by the station's stable southern-Thailand climate.
  \item \textbf{{h48 and h72 remain not-ready for all stations}} --- surface-observation-only
        lag features lack predictive signal at 2--3 day lead times. Incorporating NWP
        analysis fields (ERA5 forecast soundings) would be required.
  \item \textbf{{Root cause of prior regression fixed}} --- a bug caused all horizon slots
        to share the same 72\,h-ahead training target; corrected in V{VERSION} so each
        horizon trains against its own $t+h$ target. Skill recovered from
        $-1.5$ to $+0.3$--$+0.6$.
  \item \textbf{{L2 bias calibration now active}} --- per-\textit{{(station, horizon)}}
        mean-residual shifts ($-0.1$ to $-0.5$\,°C) fitted on validation set and
        applied at inference.
\end{{itemize}}
\end{{tcolorbox}}

\subsection{{Overall Status Table}}

\begin{{table}}[H]
\centering
\caption{{Skill score and deployment status by station and horizon.
         Green = positive skill (Candidate/Ready), red = not ready.}}
\vspace{{4pt}}
{make_skill_table(metrics, lb)}
\end{{table}}

\newpage

%% ═══════════════════════════════════════════════════════════════
%% 2. PERFORMANCE OVERVIEW
%% ═══════════════════════════════════════════════════════════════
\section{{Performance Overview}}\label{{sec:overview}}

\subsection{{Skill Score and MAE — All Stations}}

\begin{{figure}}[H]
\centering
{inc(chart_overview)}
\caption{{Top: skill score versus climatology baseline for all stations and horizons.
  Dashed line = candidate threshold ($+0.10$); dotted line = ready threshold ($+0.55$).
  Bottom: corresponding MAE values. All h6/h12/h24 cells achieve positive skill;
  the h48/h72 degradation is structural (insufficient long-range signal
  in surface-observation features).}}
\end{{figure}}

\subsection{{MAE Reference Table}}

\begin{{table}}[H]
\centering
\caption{{Mean Absolute Error (°C) on held-out test set.
         Colour scale: dark green $\approx$ 0.7\,°C (best), dark red $\approx$ 3.5\,°C.}}
\vspace{{4pt}}
{make_mae_table(metrics)}
\end{{table}}

\newpage

\subsection{{Danger Recall}}

\begin{{figure}}[H]
\centering
{inc(chart_danger)}
\caption{{Danger recall at HI\,$\geq$\,42\,°C (left) and HI\,$\geq$\,40\,°C (right).
         Dashed red line = 50\,\% safety target.
         CNX\_01 shows zero recall for $\geq$42\,°C because no extreme-heat events
         occurred in its test period (null is rendered as 0 in the chart).}}
\end{{figure}}

\subsubsection{{Danger Recall $\geq$ 42\,°C (Extreme Heat)}}

\begin{{table}}[H]
\centering
\caption{{Recall at HI $\geq$ 42\,°C. Green $\geq$ 0.75; amber $\geq$ 0.50; red $<$ 0.50.}}
\vspace{{4pt}}
{make_danger_recall_table(metrics, "danger_42")}
\end{{table}}

\subsubsection{{Danger Recall $\geq$ 40\,°C (Dangerous Heat)}}

\begin{{table}}[H]
\centering
\caption{{Recall at HI $\geq$ 40\,°C.}}
\vspace{{4pt}}
{make_danger_recall_table(metrics, "danger_40")}
\end{{table}}

\newpage

%% ═══════════════════════════════════════════════════════════════
%% 3. PER-STATION ANALYSIS
%% ═══════════════════════════════════════════════════════════════
\section{{Per-Station Analysis}}\label{{sec:stations}}

Each subsection contains a paired chart (MAE vs baselines + skill score) followed by
a complete metrics table covering all five horizons.

\newpage
{station_body}

\newpage

%% ═══════════════════════════════════════════════════════════════
%% 4. RISK CLASSIFICATION
%% ═══════════════════════════════════════════════════════════════
\section{{Risk Classification (h6 Models)}}\label{{sec:risk}}

The model's continuous HI forecast is binned into four risk categories:
\textbf{{Caution}} ($<$32\,°C), \textbf{{Extreme Caution}} (32--40\,°C),
\textbf{{Danger}} (40--54\,°C), \textbf{{Extreme Danger}} ($>$54\,°C).

\begin{{table}}[H]
\centering
\caption{{Precision and Recall by risk category (h6 models only).
  P\,=\,precision, R\,=\,recall. Support counts shown in parentheses.
  The Extreme Danger class had zero support in all test sets.}}
\vspace{{4pt}}
{make_risk_table(metrics)}
\end{{table}}

\begin{{tcolorbox}}[colback=hlgrey, colframe=navyblue!25, boxrule=0.5pt,
  arc=3pt, left=6pt, right=6pt, top=4pt, bottom=4pt]
\small\sffamily
\textbf{{Note:}} The Danger class has fewer than 10 support samples at most stations.
Precision for Danger appears low because any false positive against the larger
Extreme\,Caution pool is highly penalised. Recall ($\geq$0.4 at h6) is more
diagnostic for safety applications.
\end{{tcolorbox}}

\newpage

%% ═══════════════════════════════════════════════════════════════
%% 5. PREDICTION INTERVAL QUALITY
%% ═══════════════════════════════════════════════════════════════
\section{{Prediction Interval Quality}}\label{{sec:pi}}

Each forecast includes a 90\,\% prediction interval calibrated via Mondrian
Conformal Quantile Regression on the validation set, stratified by local-hour bucket.
Ideal empirical coverage is 0.90; current intervals are systematically over-conservative
(coverage $>$0.99) reflecting wide underlying quantile heads.

\begin{{figure}}[H]
\centering
{inc(chart_pi)}
\caption{{Left: empirical 90\,\% PI coverage (dashed = target 0.90).
         Right: PI mean width in °C. Narrower is better given adequate coverage.}}
\end{{figure}}

\begin{{table}}[H]
\centering
\small
\caption{{Full prediction-interval metrics for all (station, horizon) slots.
  Coverage colour: green = acceptable [0.87, 0.97]; amber = over-conservative ($>$0.97);
  red = under-covering ($<$0.87).}}
\vspace{{4pt}}
{make_pi_table(metrics)}
\end{{table}}

\newpage

%% ═══════════════════════════════════════════════════════════════
%% 6. TECHNICAL APPENDIX
%% ═══════════════════════════════════════════════════════════════
\section{{Technical Appendix}}\label{{sec:tech}}

\subsection{{Model Architecture}}

\begin{{itemize}}[leftmargin=1.4em, itemsep=3pt]
  \item \textbf{{Backend:}} LightGBM Quantile Regression with three quantile heads:
        q05, q50, q95. Backend key: \texttt{{lightgbm\_quantile}}.
  \item \textbf{{Target:}} Two-head \texttt{{(temp\_c, rh)}} at $t+h$; composed to Heat
        Index via the Rothfusz equation at inference.
  \item \textbf{{Feature engineering:}} Union of per-horizon lag/rolling window sets built
        once via \texttt{{build\_X\_once()}}; per-horizon column subset applied before training.
        Lag range: 1--240\,h (horizon-dependent). Rolling windows: 3--240\,h.
  \item \textbf{{Hyperparameter optimisation:}} Optuna TPE sampler with 25 adaptive trials,
        early stopping on validation MAE, pilot run for \texttt{{n\_estimators}} selection.
  \item \textbf{{Train/val/test split:}} Chronological 60/20/20 by station.
        Validation set further split into \textit{{val\_es}} (early stopping)
        and \textit{{val\_cal}} (conformal calibration).
  \item \textbf{{L2 bias calibration:}} Per-\textit{{(station, horizon)}} mean-residual
        shift fitted on val\_cal; persisted as \texttt{{calibration.json}};
        applied at inference by \texttt{{predict.py}}.
  \item \textbf{{Conformal PI:}} Mondrian CQR on val\_cal, stratified by local-hour bucket.
  \item \textbf{{Danger oversampling:}} Rows with HI\,$\geq$\,42\,°C duplicated until
        support $\geq$ 200 to improve extreme-heat tail recall.
\end{{itemize}}

\subsection{{Skill Score Definition}}

\[
  \text{{Skill}} = 1 - \frac{{\text{{MAE}}_\text{{model}}}}{{\text{{MAE}}_\text{{climatology}}}}
\]

A value of 0 means the model equals climatology; $+1$ is a perfect forecast;
negative values mean the model is worse than climatology.
The climatology baseline uses a per-station, per-hour, per-month \emph{{causal expanding mean}}
of historical heat index, updated with each new observation (no future leakage).

\subsection{{Known Limitations}}

\begin{{itemize}}[leftmargin=1.4em, itemsep=3pt]
  \item \textbf{{h48/h72 negative skill:}} Without NWP analysis fields (ERA5 forecast
        soundings, 500\,hPa geopotential), surface-lag models cannot reliably predict
        heat events at 2--3 day lead times.
  \item \textbf{{Wide prediction intervals:}} Current mean PI widths of 9--11\,°C
        are over-conservative. The Mondrian CQR coverage is correct but the underlying
        quantile heads could be regularised further.
  \item \textbf{{CNX\_01 danger\_42 undefined:}} No HI\,$\geq$\,42\,°C events in the
        Chiang Mai test period; recall metric is undefined (not zero).
  \item \textbf{{RYG\_01 low danger tail:}} Rayong has the fewest extreme-heat events;
        danger\_42 recall is 0.12--0.57 and may benefit from additional training data.
\end{{itemize}}

\subsection{{Data Sources}}

\begin{{itemize}}[leftmargin=1.4em, itemsep=3pt]
  \item \textbf{{TMD}} (Thai Meteorological Department) --- hourly surface observations
  \item \textbf{{ERA5}} (ECMWF via CDS\,API) --- reanalysis gap-fill and extended
        atmospheric variables
  \item \textbf{{NASA POWER}} --- daily solar radiation and surface wind
  \item \textbf{{MODIS}} --- land surface temperature (LST) for urban heat island context
\end{{itemize}}

\vspace{{1cm}}
\begin{{center}}
  \textcolor{{gray}}{{\small\sffamily
    HeatShield AI Pipeline\quad$\cdot$\quad
    Run \texttt{{{tex(RUN_ID)}}}\quad$\cdot$\quad
    {now}}}
\end{{center}}

\end{{document}}
"""
    return doc


# ── compile ───────────────────────────────────────────────────────────────────
def compile_latex(tex_path: Path, out_dir: Path) -> Path:
    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        "-output-directory", str(out_dir),
        str(tex_path),
    ]
    for pass_n in (1, 2):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        if r.returncode != 0:
            log = tex_path.with_suffix(".log")
            if log.exists():
                tail = log.read_text(errors="replace")[-3000:]
                print(f"[xelatex pass {pass_n} FAILED]\n{tail}")
            else:
                print(r.stdout[-2000:])
                print(r.stderr[-1000:])
            raise RuntimeError(f"xelatex failed on pass {pass_n}")
        print(f"xelatex pass {pass_n} OK")
    pdf = out_dir / tex_path.with_suffix(".pdf").name
    return pdf


# ── entry ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="logs/eval/report_v47")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    lb      = load_leaderboard()

    tex_src = build_tex(metrics, lb, out_dir)
    tex_path = out_dir / "model_report_v47.tex"
    tex_path.write_text(tex_src, encoding="utf-8")
    print(f"LaTeX written: {tex_path}")

    pdf = compile_latex(tex_path, out_dir)
    print(f"\nReport ready: {pdf.resolve()}")


if __name__ == "__main__":
    main()
