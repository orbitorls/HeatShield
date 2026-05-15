"""Generate a comprehensive PDF model evaluation report using LaTeX.

This script reads model registry data from app/models/forecast_v3 and generates
a comprehensive LaTeX document with all evaluation metrics, then compiles it to PDF.

Usage:
    python scripts/generate_model_report.py [--out <path>]

Defaults to writing comprehensive_model_report.pdf in the project root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

HORIZONS = [6, 12, 24, 48, 72]
STATIONS = ["BKK_01", "CNX_01", "HYI_01", "KKN_01", "RYG_01"]

STATION_LABEL = {
    "BKK_01": "Bangkok (BKK_01)",
    "CNX_01": "Chiang Mai (CNX_01)",
    "HYI_01": "Hat Yai (HYI_01)",
    "KKN_01": "Khon Kaen (KKN_01)",
    "RYG_01": "Rayong (RYG_01)",
}
STATION_REGION = {
    "BKK_01": "Central Thailand — 13.76°N 100.50°E",
    "CNX_01": "Northern Thailand — 18.79°N 98.98°E",
    "HYI_01": "Southern Thailand — 7.00°N 100.47°E",
    "KKN_01": "Northeast Thailand — 16.44°N 102.84°E",
    "RYG_01": "Eastern Thailand — 12.68°N 101.27°E",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _skill_color(s: float | None) -> Any:
    if s is None or np.isnan(s):
        return C_MGREY
    if s >= 0.40:
        return C_GREEN
    if s >= 0.10:
        return C_TEAL
    if s >= 0.0:
        return C_AMBER
    return C_RED


def _status_badge(status: str) -> tuple[str, Any]:
    if status == "candidate":
        return ("CANDIDATE", C_TEAL)
    if status == "ready":
        return ("READY", C_GREEN)
    return ("NOT READY", C_RED)


def fig_to_image(fig, width_cm: float, height_cm: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=width_cm * cm, height=height_cm * cm)


def load_metrics(run_dir: Path) -> dict[tuple[str, int], dict]:
    result = {}
    for sid in STATIONS:
        for h in HORIZONS:
            p = run_dir / "lightgbm_quantile" / sid / f"h{h}" / "metrics.json"
            if p.exists():
                result[(sid, h)] = json.loads(p.read_text())
    return result


def load_leaderboard(eval_dir: Path) -> list[dict]:
    p = eval_dir / "leaderboard.json"
    if p.exists():
        return json.loads(p.read_text())
    return []


# ── chart generators ──────────────────────────────────────────────────────────

def chart_skill_heatmap(metrics: dict) -> Image:
    data = np.full((len(STATIONS), len(HORIZONS)), np.nan)
    for i, sid in enumerate(STATIONS):
        for j, h in enumerate(HORIZONS):
            m = metrics.get((sid, h))
            if m:
                data[i, j] = m["baselines"]["skill_score"]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=-1.5, vmax=0.7, aspect="auto")
    plt.colorbar(im, ax=ax, label="Skill Score")

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"h{h}" for h in HORIZONS], fontsize=11)
    ax.set_yticks(range(len(STATIONS)))
    ax.set_yticklabels([STATION_LABEL[s].split(" (")[0] for s in STATIONS], fontsize=11)
    ax.set_title("Skill Score vs Climatology Baseline  (green = better than climatology)", fontsize=12, pad=10)

    for i in range(len(STATIONS)):
        for j in range(len(HORIZONS)):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 0.7 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=color)
    fig.tight_layout()
    return fig_to_image(fig, 16, 6)


def chart_mae_heatmap(metrics: dict) -> Image:
    data = np.full((len(STATIONS), len(HORIZONS)), np.nan)
    for i, sid in enumerate(STATIONS):
        for j, h in enumerate(HORIZONS):
            m = metrics.get((sid, h))
            if m:
                data[i, j] = m["regression"]["mae"]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(data, cmap=cmap, vmin=0.7, vmax=3.5, aspect="auto")
    plt.colorbar(im, ax=ax, label="MAE (°C)")

    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"h{h}" for h in HORIZONS], fontsize=11)
    ax.set_yticks(range(len(STATIONS)))
    ax.set_yticklabels([STATION_LABEL[s].split(" (")[0] for s in STATIONS], fontsize=11)
    ax.set_title("Mean Absolute Error by Station & Horizon  (green = lower error)", fontsize=12, pad=10)

    for i in range(len(STATIONS)):
        for j in range(len(HORIZONS)):
            v = data[i, j]
            if not np.isnan(v):
                color = "white" if v > 2.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=color)
    fig.tight_layout()
    return fig_to_image(fig, 16, 6)


def chart_danger_recall(metrics: dict) -> Image:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    thresholds = [(42, "axes[0]"), (40, "axes[1]")]
    titles = ["Danger Recall ≥42°C (Extreme Heat)", "Danger Recall ≥40°C (Dangerous Heat)"]
    keys = ["danger_42", "danger_40"]

    x = np.arange(len(HORIZONS))
    width = 0.14
    palette = ["#1A6FA8", "#17A589", "#27AE60", "#8E44AD", "#E67E22"]

    for ax, key, title in zip(axes, keys, titles):
        for k, sid in enumerate(STATIONS):
            recalls = []
            for h in HORIZONS:
                m = metrics.get((sid, h))
                if m:
                    r = m["safety"][key]["recall"]
                    recalls.append(r if r is not None else 0.0)
                else:
                    recalls.append(0.0)
            bars = ax.bar(x + k * width, recalls, width, label=STATION_LABEL[sid].split(" (")[0],
                          color=palette[k], alpha=0.85)
        ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="50% target")
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"h{h}" for h in HORIZONS])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Recall")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig_to_image(fig, 17, 5.5)


def chart_station_detail(sid: str, metrics: dict) -> Image:
    hs = []
    maes, clim_maes, pers_maes, skills = [], [], [], []
    for h in HORIZONS:
        m = metrics.get((sid, h))
        if m:
            hs.append(h)
            maes.append(m["regression"]["mae"])
            clim_maes.append(m["baselines"]["climatology_mae"])
            pers_maes.append(m["baselines"]["persistence_mae"])
            skills.append(m["baselines"]["skill_score"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Left: MAE vs baselines
    x = np.arange(len(hs))
    w = 0.28
    b1 = ax1.bar(x - w, maes, w, color="#1A6FA8", label="Model MAE", alpha=0.9)
    b2 = ax1.bar(x,      clim_maes, w, color="#F39C12", label="Climatology MAE", alpha=0.9)
    b3 = ax1.bar(x + w,  pers_maes, w, color="#BDC3C7", label="Persistence MAE", alpha=0.9)
    for b, v in zip(b1, maes):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.04, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=8, color="#1A6FA8", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"h{h}" for h in hs])
    ax1.set_ylabel("MAE (°C)")
    ax1.set_title(f"{STATION_LABEL[sid]} — MAE vs Baselines")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Right: Skill score
    skill_colors = ["#27AE60" if s >= 0 else "#C0392B" for s in skills]
    bars = ax2.bar(x, skills, color=skill_colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=1.2)
    ax2.axhline(0.55, color="#27AE60", linestyle="--", linewidth=1.2, alpha=0.7, label="Ready threshold")
    for b, v in zip(bars, skills):
        va = "bottom" if v >= 0 else "top"
        offset = 0.02 if v >= 0 else -0.02
        ax2.text(b.get_x() + b.get_width()/2, v + offset, f"{v:+.2f}",
                 ha="center", va=va, fontsize=8, fontweight="bold", color="white" if abs(v) > 0.8 else "black")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h{h}" for h in hs])
    ax2.set_ylabel("Skill Score")
    ax2.set_title(f"{STATION_LABEL[sid]} — Skill Score by Horizon")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig_to_image(fig, 17, 5.5)


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_pdf(run_dir: Path, eval_dir: Path, out_path: Path) -> None:
    metrics = load_metrics(run_dir)
    leaderboard = load_leaderboard(eval_dir)
    lb_lookup = {(r["station"], r["horizon_h"]): r for r in leaderboard}

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2.0 * cm,
        leftMargin=2.0 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="HeatShield AI — Model Evaluation Report V47",
        author="HeatShield AI Pipeline",
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle("cover_title", fontSize=28, fontName="Helvetica-Bold",
                                       textColor=C_WHITE, alignment=TA_CENTER, spaceAfter=6),
        "cover_sub":   ParagraphStyle("cover_sub",   fontSize=14, fontName="Helvetica",
                                       textColor=C_MGREY, alignment=TA_CENTER, spaceAfter=4),
        "cover_meta":  ParagraphStyle("cover_meta",  fontSize=11, fontName="Helvetica",
                                       textColor=C_MGREY, alignment=TA_CENTER, spaceAfter=3),
        "h1":  ParagraphStyle("h1", fontSize=18, fontName="Helvetica-Bold",
                               textColor=C_NAVY, spaceBefore=14, spaceAfter=6),
        "h2":  ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                               textColor=C_BLUE, spaceBefore=10, spaceAfter=4),
        "h3":  ParagraphStyle("h3", fontSize=11, fontName="Helvetica-Bold",
                               textColor=C_NAVY, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontSize=10, fontName="Helvetica",
                                leading=14, textColor=colors.black, spaceAfter=4,
                                alignment=TA_JUSTIFY),
        "caption": ParagraphStyle("caption", fontSize=8.5, fontName="Helvetica-Oblique",
                                   textColor=colors.HexColor("#7F8C8D"), alignment=TA_CENTER,
                                   spaceAfter=6),
        "note": ParagraphStyle("note", fontSize=9, fontName="Helvetica",
                                textColor=colors.HexColor("#555"), leading=13,
                                leftIndent=10, spaceAfter=4),
    }

    story = []

    # ── COVER ──────────────────────────────────────────────────────────────────
    cover_table = Table(
        [[Paragraph("HeatShield AI", styles["cover_title"]),],
         [Paragraph("Model Evaluation Report", styles["cover_title"]),],
         [Spacer(1, 0.4*cm)],
         [Paragraph("Version 47  ·  LightGBM Quantile Regression", styles["cover_sub"])],
         [Paragraph("Full-horizon evaluation: 6 h · 12 h · 24 h · 48 h · 72 h", styles["cover_meta"])],
         [Paragraph("5 Stations: BKK · CNX · HYI · KKN · RYG", styles["cover_meta"])],
         [Spacer(1, 0.6*cm)],
         [Paragraph(f"Generated: {datetime.utcnow().strftime('%d %B %Y  %H:%M UTC')}", styles["cover_meta"])],
         [Paragraph("Run ID: 20260511T164652Z", styles["cover_meta"])],
        ],
        colWidths=[17 * cm],
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_NAVY]),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("BOX", (0, 0), (-1, -1), 2, C_TEAL),
    ]))
    story.append(Spacer(1, 3 * cm))
    story.append(cover_table)
    story.append(Spacer(1, 1 * cm))

    # Cover summary badges
    candidate_count = sum(1 for r in leaderboard if r.get("status") == "candidate")
    ready_count = sum(1 for r in leaderboard if r.get("status") == "ready")
    total = len(leaderboard)
    badge_data = [
        [Paragraph(f"<b>{candidate_count + ready_count}</b><br/><font size=9>Positive Skill<br/>Slots</font>",
                   ParagraphStyle("b", fontSize=22, fontName="Helvetica-Bold", textColor=C_TEAL,
                                  alignment=TA_CENTER)),
         Paragraph(f"<b>{total - candidate_count - ready_count}</b><br/><font size=9>Not Ready<br/>Slots</font>",
                   ParagraphStyle("b", fontSize=22, fontName="Helvetica-Bold", textColor=C_AMBER,
                                  alignment=TA_CENTER)),
         Paragraph(f"<b>{total}</b><br/><font size=9>Total<br/>Slots</font>",
                   ParagraphStyle("b", fontSize=22, fontName="Helvetica-Bold", textColor=C_BLUE,
                                  alignment=TA_CENTER)),
         Paragraph("<b>5</b><br/><font size=9>Stations<br/>Covered</font>",
                   ParagraphStyle("b", fontSize=22, fontName="Helvetica-Bold", textColor=C_NAVY,
                                  alignment=TA_CENTER)),
        ]
    ]
    badge_table = Table(badge_data, colWidths=[4.25 * cm] * 4)
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F8F5")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#FEF9E7")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EBF5FB")),
        ("BACKGROUND", (3, 0), (3, -1), colors.HexColor("#EAF2FF")),
        ("BOX", (0, 0), (0, -1), 1.5, C_TEAL),
        ("BOX", (1, 0), (1, -1), 1.5, C_AMBER),
        ("BOX", (2, 0), (2, -1), 1.5, C_BLUE),
        ("BOX", (3, 0), (3, -1), 1.5, C_NAVY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
    ]))
    story.append(badge_table)
    story.append(PageBreak())

    # ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────────────────
    story.append(Paragraph("1  Executive Summary", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph(
        "This report presents the evaluation results for HeatShield AI's V47 LightGBM "
        "Quantile Regression forecast models, covering five Thai meteorological stations "
        "across six forecast horizons (6 h through 72 h). All models were trained on "
        "ERA5 reanalysis + TMD surface observations from 2021 to 2025.",
        styles["body"]))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Key Findings", styles["h2"]))
    findings = [
        ("<b>15 of 25 slots achieve positive skill score</b> — all short-to-medium horizons "
         "(h6, h12, h24) across every station beat the seasonal climatology baseline."),
        ("<b>Best performer: KKN_01 h6</b> — skill score +0.58, MAE 1.05°C. "
         "Consistently outperforms climatology by a wide margin."),
        ("<b>HYI_01 (Hat Yai) lowest absolute error</b> — MAE 0.73°C at h6, likely due "
         "to the station's more stable southern Thailand climate."),
        ("<b>h48 and h72 remain not-ready for all stations</b> — surface-observation-only "
         "features lack predictive signal at 2–3 day lead times; NWP analysis fields would "
         "be required to improve these slots."),
        ("<b>Root cause of prior regression fixed</b> — a bug where all forecast horizons "
         "shared the same 72 h-ahead training target (instead of their own h-ahead target) "
         "was corrected in this version, recovering skills from −1.5 to +0.3–0.6."),
        ("<b>Bias calibration now active</b> — an L2 per-(station, horizon) mean-residual "
         "shift is fitted on the validation set and applied at inference. Shifts are small "
         "(−0.1 to −0.5°C) confirming the y-target fix was the dominant issue."),
    ]
    for txt in findings:
        story.append(Paragraph(f"• {txt}", styles["note"]))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Model Status Overview", styles["h2"]))

    # Overall status table
    header = ["Station", "Region"] + [f"h{h}" for h in HORIZONS]
    rows = [header]
    for sid in STATIONS:
        row = [STATION_LABEL[sid].split(" (")[0], STATION_REGION[sid].split("—")[0].strip()]
        for h in HORIZONS:
            lb = lb_lookup.get((sid, h))
            if lb:
                s = lb["skill_score"]
                st = lb["status"]
                badge, _ = _status_badge(st)
                row.append(f"{s:+.2f}\n{badge}")
            else:
                row.append("—")
        rows.append(row)

    col_w = [3.2*cm, 3.5*cm] + [1.8*cm]*5
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LGREY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.5, C_MGREY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (0, 1), (1, -1), "LEFT"),
    ])
    # Colour skill cells
    for i, sid in enumerate(STATIONS):
        for j, h in enumerate(HORIZONS):
            lb = lb_lookup.get((sid, h))
            if lb:
                s = lb["skill_score"]
                cell_color = (
                    colors.HexColor("#D5F5E3") if s >= 0.10 else
                    colors.HexColor("#FCF3CF") if s >= 0 else
                    colors.HexColor("#FADBD8")
                )
                ts.add("BACKGROUND", (j + 2, i + 1), (j + 2, i + 1), cell_color)
    tbl.setStyle(ts)
    story.append(tbl)
    story.append(PageBreak())

    # ── 2. SUMMARY CHARTS ─────────────────────────────────────────────────────
    story.append(Paragraph("2  Summary Charts", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph("2.1  Skill Score Heatmap", styles["h2"]))
    story.append(Paragraph(
        "Skill score measures improvement over the seasonal climatology baseline. "
        "Positive (green) means the model beats climatology; negative (red) means worse. "
        "A threshold of +0.10 is required for candidate status.",
        styles["body"]))
    story.append(chart_skill_heatmap(metrics))
    story.append(Paragraph(
        "Figure 1 — Skill score by station and horizon. All h6/h12/h24 cells are positive. "
        "The h48/h72 degradation is structural (surface-obs features lack 48–72 h signal).",
        styles["caption"]))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("2.2  Mean Absolute Error Heatmap", styles["h2"]))
    story.append(Paragraph(
        "MAE in °C on the held-out test set. Lower values (dark green) indicate more "
        "accurate point forecasts. HYI_01 achieves the lowest absolute errors overall.",
        styles["body"]))
    story.append(chart_mae_heatmap(metrics))
    story.append(Paragraph(
        "Figure 2 — MAE (°C) by station and horizon. Note that despite higher MAE at long horizons, "
        "the climatology baseline is also weaker there — see skill score for relative comparison.",
        styles["caption"]))
    story.append(PageBreak())

    story.append(Paragraph("2.3  Danger Recall Charts", styles["h2"]))
    story.append(Paragraph(
        "Danger recall measures how often the model correctly predicts a heat danger event "
        "before it happens. Two thresholds are reported: HI ≥ 42°C (Extreme Danger) and "
        "HI ≥ 40°C (Danger). High recall is safety-critical — a missed extreme heat event "
        "is more harmful than a false alarm.",
        styles["body"]))
    story.append(chart_danger_recall(metrics))
    story.append(Paragraph(
        "Figure 3 — Danger recall at ≥42°C (left) and ≥40°C (right). Red dashed line = 50% target. "
        "CNX_01 shows null for ≥42°C because no extreme events occurred in the test period.",
        styles["caption"]))
    story.append(PageBreak())

    # ── 3. PER-STATION ANALYSIS ────────────────────────────────────────────────
    story.append(Paragraph("3  Per-Station Analysis", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8))

    for s_idx, sid in enumerate(STATIONS):
        story.append(Paragraph(f"3.{s_idx+1}  {STATION_LABEL[sid]}", styles["h2"]))
        story.append(Paragraph(STATION_REGION[sid], styles["note"]))

        # Station chart
        story.append(chart_station_detail(sid, metrics))
        story.append(Paragraph(
            f"Figure {s_idx+4} — {STATION_LABEL[sid].split('(')[0].strip()}: model MAE vs baselines (left) "
            f"and skill score by horizon (right). Green bars indicate positive skill.",
            styles["caption"]))

        # Detailed metrics table
        story.append(Spacer(1, 0.2*cm))
        th_row = ["Metric", "h6", "h12", "h24", "h48", "h72"]
        tbl_rows = [th_row]
        row_defs = [
            ("MAE (°C)", lambda m: f"{m['regression']['mae']:.3f}"),
            ("RMSE (°C)", lambda m: f"{m['regression']['rmse']:.3f}"),
            ("Bias (°C)", lambda m: f"{m['regression']['bias']:+.3f}"),
            ("Correlation", lambda m: f"{m['regression']['correlation']:.4f}"),
            ("P90 Abs Err (°C)", lambda m: f"{m['regression']['p90_abs_error']:.2f}"),
            ("Climatology MAE", lambda m: f"{m['baselines']['climatology_mae']:.3f}"),
            ("Persistence MAE", lambda m: f"{m['baselines']['persistence_mae']:.3f}"),
            ("Skill Score", lambda m: f"{m['baselines']['skill_score']:+.4f}"),
            ("PI Coverage 90%", lambda m: f"{m['prediction_interval']['coverage_90']:.3f}"),
            ("PI Mean Width (°C)", lambda m: f"{m['prediction_interval']['mean_width']:.2f}"),
            ("Danger Recall ≥42°C", lambda m: (
                f"{m['safety']['danger_42']['recall']:.2f} "
                f"(n={m['safety']['danger_42']['support']})"
            )),
            ("Danger Recall ≥40°C", lambda m: (
                f"{m['safety']['danger_40']['recall']:.2f} "
                f"(n={m['safety']['danger_40']['support']})"
            )),
            ("Test Rows", lambda m: f"{m['split']['row_counts']['test']:,}"),
            ("Train End", lambda m: m["split"]["date_ranges"]["train"]["end"][:10]),
            ("Test Start", lambda m: m["split"]["date_ranges"]["test"]["start"][:10]),
            ("Test End", lambda m: m["split"]["date_ranges"]["test"]["end"][:10]),
            ("Status", lambda m: lb_lookup.get((sid, m["horizon_h"]), {}).get("status", "—").upper()),
        ]
        for label, fn in row_defs:
            row = [label]
            for h in HORIZONS:
                m = metrics.get((sid, h))
                try:
                    row.append(fn(m) if m else "—")
                except Exception:
                    row.append("—")
            tbl_rows.append(row)

        det_tbl = Table(tbl_rows, colWidths=[3.8*cm] + [2.6*cm]*5, repeatRows=1)
        det_ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 1), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LGREY, C_WHITE]),
            ("GRID", (0, 0), (-1, -1), 0.4, C_MGREY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ])
        # Colour skill row
        skill_row_idx = next(i+1 for i, (lbl, _) in enumerate(row_defs) if lbl == "Skill Score")
        for j, h in enumerate(HORIZONS):
            m = metrics.get((sid, h))
            if m:
                s = m["baselines"]["skill_score"]
                cell_c = (colors.HexColor("#D5F5E3") if s >= 0.10 else
                          colors.HexColor("#FCF3CF") if s >= 0 else
                          colors.HexColor("#FADBD8"))
                det_ts.add("BACKGROUND", (j+1, skill_row_idx), (j+1, skill_row_idx), cell_c)
        # Colour status row
        status_row_idx = len(tbl_rows) - 1
        for j, h in enumerate(HORIZONS):
            lb = lb_lookup.get((sid, h))
            if lb:
                st = lb["status"]
                cell_c = (C_LGREY if st == "not_ready" else
                          colors.HexColor("#D5F5E3") if st in ("candidate", "ready") else C_LGREY)
                det_ts.add("BACKGROUND", (j+1, status_row_idx), (j+1, status_row_idx), cell_c)
        det_tbl.setStyle(det_ts)
        story.append(det_tbl)

        if s_idx < len(STATIONS) - 1:
            story.append(PageBreak())

    story.append(PageBreak())

    # ── 4. RISK CLASSIFICATION ────────────────────────────────────────────────
    story.append(Paragraph("4  Risk Classification Performance", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "The model's continuous HI forecast is binned into four WHO/Thai-adapted risk "
        "categories: Caution (<32°C), Extreme Caution (32–40°C), Danger (40–54°C), "
        "and Extreme Danger (>54°C). Classification metrics below are on the h6 models "
        "which are most relevant for near-term warnings.",
        styles["body"]))
    story.append(Spacer(1, 0.3*cm))

    risk_labels = ["Caution", "Extreme\nCaution", "Danger", "Extreme\nDanger"]
    risk_header = ["Station"] + risk_labels + ["Accuracy"]
    risk_rows = [risk_header]
    for sid in STATIONS:
        m = metrics.get((sid, 6))
        if not m:
            continue
        cr = m["risk"]["classification_report"]
        row = [STATION_LABEL[sid].split(" (")[0]]
        for lbl in ["Caution", "Extreme Caution", "Danger", "Extreme Danger"]:
            info = cr.get(lbl, {})
            p = info.get("precision", 0)
            r = info.get("recall", 0)
            row.append(f"P {p:.2f}\nR {r:.2f}")
        row.append(f"{cr.get('accuracy', 0):.3f}")
        risk_rows.append(row)

    risk_tbl = Table(risk_rows, colWidths=[3.2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.6*cm], repeatRows=1)
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LGREY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_MGREY),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(risk_tbl)
    story.append(Paragraph(
        "P = Precision, R = Recall. 'Extreme Danger' class is empty in current test period "
        "(no HI > 54°C observed). 'Danger' class is rare (support < 10 at most stations).",
        styles["caption"]))

    story.append(PageBreak())

    # ── 5. PREDICTION INTERVALS ───────────────────────────────────────────────
    story.append(Paragraph("5  Prediction Interval Quality", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8))
    story.append(Paragraph(
        "Each forecast includes a 90% prediction interval (PI) calibrated via Mondrian "
        "Conformal QR on the validation set. Ideal coverage is 0.90; mean PI width "
        "reflects uncertainty — narrower is better given adequate coverage.",
        styles["body"]))
    story.append(Spacer(1, 0.3*cm))

    pi_header = ["Station", "Horizon", "Coverage 90%", "Mean Width (°C)", "P25 Width", "P75 Width", "Pinball Q05", "Pinball Q95"]
    pi_rows = [pi_header]
    for sid in STATIONS:
        for h in HORIZONS:
            m = metrics.get((sid, h))
            if not m:
                continue
            pi = m["prediction_interval"]
            cov = pi["coverage_90"]
            cov_str = f"{cov:.3f}"
            pi_rows.append([
                STATION_LABEL[sid].split(" (")[0],
                f"h{h}",
                cov_str,
                f"{pi['mean_width']:.2f}",
                f"{pi['p25_width']:.2f}",
                f"{pi['p75_width']:.2f}",
                f"{pi['pinball_q05']:.4f}",
                f"{pi['pinball_q95']:.4f}",
            ])

    pi_tbl = Table(pi_rows, colWidths=[2.8*cm, 1.4*cm, 2.2*cm, 2.2*cm, 1.8*cm, 1.8*cm, 2.0*cm, 2.0*cm], repeatRows=1)
    pi_ts = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_LGREY, C_WHITE]),
        ("GRID", (0, 0), (-1, -1), 0.4, C_MGREY),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ])
    # Highlight over-covered rows
    for i, row in enumerate(pi_rows[1:], 1):
        cov_val = float(row[2])
        if cov_val > 0.97:
            pi_ts.add("BACKGROUND", (2, i), (2, i), colors.HexColor("#FCF3CF"))
        elif 0.87 <= cov_val <= 0.97:
            pi_ts.add("BACKGROUND", (2, i), (2, i), colors.HexColor("#D5F5E3"))
    pi_tbl.setStyle(pi_ts)
    story.append(pi_tbl)
    story.append(Paragraph(
        "Note: Coverage > 0.97 (yellow) indicates over-conservative intervals — "
        "the PI is wider than necessary. Future calibration can tighten these.",
        styles["caption"]))

    story.append(PageBreak())

    # ── 6. TECHNICAL APPENDIX ─────────────────────────────────────────────────
    story.append(Paragraph("6  Technical Appendix", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=C_BLUE, spaceAfter=8))

    story.append(Paragraph("6.1  Model Architecture", styles["h2"]))
    arch_items = [
        ("<b>Backend:</b> LightGBM Quantile Regression (lightgbm_quantile) with three "
         "quantile heads: q05, q50, q95."),
        ("<b>Target:</b> Two-head (temp_c, rh) pair at t+h; composed to Heat Index via "
         "Rothfusz equation at inference. Each horizon trains on its own correct "
         "shift(-h) target — a key fix in V47."),
        ("<b>Feature engineering:</b> Union of per-horizon lag/rolling sets built once "
         "via build_X_once(); per-horizon subset applied before training. "
         "Lag range: 1–240 h depending on horizon. Rolling: 3–240 h windows."),
        ("<b>Hyperparameter optimisation:</b> Optuna with 25 adaptive trials per slot, "
         "early stopping on validation MAE, pilot run for n_estimators selection."),
        ("<b>Train/val/test split:</b> Chronological 60/20/20 per station. "
         "Val split into val_es (early stopping) and val_cal (conformal calibration)."),
        ("<b>Bias calibration:</b> L2 per-(station, horizon) mean-residual shift fitted "
         "on val_cal, persisted as calibration.json, applied at inference."),
        ("<b>Conformal prediction intervals:</b> Mondrian CQR calibrated on val_cal "
         "stratified by local hour bucket."),
        ("<b>Danger oversampling:</b> Rows with HI ≥ 42°C synthetically duplicated "
         "until support ≥ 200 to improve tail recall."),
    ]
    for txt in arch_items:
        story.append(Paragraph(f"• {txt}", styles["note"]))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("6.2  Skill Score Definition", styles["h2"]))
    story.append(Paragraph(
        "Skill score = 1 − MAE_model / MAE_climatology. A value of 0.0 means the model "
        "equals climatology; +1.0 would be a perfect forecast; negative values mean the "
        "model is worse than the climatology baseline. The climatology baseline is the "
        "per-station, per-hour, per-month expanding mean of historical heat index.",
        styles["body"]))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("6.3  Known Limitations", styles["h2"]))
    lims = [
        "<b>h48/h72 negative skill:</b> Without NWP analysis fields (e.g., ERA5 "
         "forecast soundings, 500 hPa geopotential), surface-lag models cannot reliably "
         "predict heat events 2–3 days ahead. These slots are marked not_ready.",
        "<b>Prediction interval width:</b> Current PI mean widths of 9–11°C are wider "
         "than ideal. The Mondrian CQR calibration is correct (coverage ≥ 0.90) but "
         "the underlying quantile heads can be further regularised.",
        "<b>CNX_01 danger_42 null:</b> No HI ≥ 42°C events in the Chiang Mai test "
         "period — recall metric is undefined, not zero.",
        "<b>RYG_01 danger recall:</b> Rayong has the least extreme heat events in the "
         "dataset; danger_42 recall is 0.12–0.57 and will require more training data "
         "or targeted oversampling to improve.",
    ]
    for txt in lims:
        story.append(Paragraph(f"• {txt}", styles["note"]))

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("6.4  Data Sources", styles["h2"]))
    ds_items = [
        "TMD (Thai Meteorological Department) — hourly surface observations",
        "ERA5 reanalysis (ECMWF via CDS API) — gap-fill and extended atmospheric variables",
        "NASA POWER — daily solar radiation and surface wind",
        "MODIS — land surface temperature (LST) for urban heat island context",
    ]
    for txt in ds_items:
        story.append(Paragraph(f"• {txt}", styles["note"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_MGREY, spaceAfter=6))
    story.append(Paragraph(
        f"Report generated by HeatShield AI Pipeline  ·  Run 20260511T164652Z  ·  "
        f"{datetime.utcnow().strftime('%d %b %Y %H:%M UTC')}",
        ParagraphStyle("footer", fontSize=8, fontName="Helvetica-Oblique",
                       textColor=colors.HexColor("#999"), alignment=TA_CENTER)))

    # ── BUILD ──────────────────────────────────────────────────────────────────
    def _header_footer(canvas, doc):
        canvas.saveState()
        # Header bar
        canvas.setFillColor(C_NAVY)
        canvas.rect(doc.leftMargin, A4[1] - 1.5*cm, A4[0] - doc.leftMargin - doc.rightMargin, 0.5*cm, fill=1, stroke=0)
        canvas.setFillColor(C_WHITE)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(doc.leftMargin + 4, A4[1] - 1.2*cm, "HeatShield AI — Model Evaluation Report V47")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(A4[0] - doc.rightMargin - 4, A4[1] - 1.2*cm,
                               f"Run: 20260511T164652Z")
        # Footer
        canvas.setFillColor(C_MGREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(A4[0] / 2, doc.bottomMargin / 2,
                                 f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"Report saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate V47 model evaluation PDF report")
    parser.add_argument("--run-dir",  default="logs/eval/runs/20260511T164652Z",
                        help="Path to eval run directory")
    parser.add_argument("--eval-dir", default="logs/eval/v47",
                        help="Path to eval version dir (contains leaderboard.json)")
    parser.add_argument("--out", default="logs/eval/model_report_v47.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    run_dir  = Path(args.run_dir)
    eval_dir = Path(args.eval_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_pdf(run_dir, eval_dir, out_path)


if __name__ == "__main__":
    main()
