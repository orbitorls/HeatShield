"""Multi-page PDF evaluation report generator using fpdf2."""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF
from fpdf.enums import XPos, YPos

_CATEGORY_LABELS = ["Caution", "Extreme Caution", "Danger", "Extreme Danger"]


def _hours_from_features(X: pd.DataFrame, n: int) -> np.ndarray:
    if {"hour_sin", "hour_cos"} <= set(X.columns):
        return (np.round(np.arctan2(X["hour_sin"], X["hour_cos"]) / (2 * np.pi / 24)) % 24).astype(int).to_numpy()
    return np.arange(n) % 24


def _station_names(X: pd.DataFrame, labels: Mapping[int, str] | None) -> pd.Series:
    if "station_enc" not in X.columns:
        return pd.Series(["all"] * len(X))
    codes = X["station_enc"].astype(int)
    if labels:
        return codes.map(lambda c: labels.get(int(c), f"station_{int(c)}"))
    return codes.map(lambda c: f"station_{int(c)}")


def _save_confusion(metrics: dict, out_dir: Path, *, dpi: int = 80) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    sns.heatmap(metrics["risk"]["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=_CATEGORY_LABELS, yticklabels=_CATEGORY_LABELS, ax=axes[0])
    axes[0].set_title("Risk Category Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    sns.heatmap(metrics["risk"]["confusion_matrix_normalized"], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=_CATEGORY_LABELS, yticklabels=_CATEGORY_LABELS, ax=axes[1])
    axes[1].set_title("Risk Category Recall")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_classification(
    metrics: dict, out_dir: Path, horizon_h: int = 24, station: str = "", *, dpi: int = 80
) -> None:
    report = metrics["risk"]["classification_report"]
    rows = [report[label] for label in _CATEGORY_LABELS]
    x = np.arange(len(_CATEGORY_LABELS))
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, key in [(-0.25, "precision"), (0.0, "recall"), (0.25, "f1-score")]:
        ax.bar(x + offset, [r[key] for r in rows], width=0.25, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(_CATEGORY_LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Risk Category")
    ax.set_ylabel("Score")
    ax.set_title(f"Classification Report | h{horizon_h} | {station}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_report.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_error_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    out_dir: Path,
    station_labels: Mapping[int, str] | None,
    horizon_h: int = 24,
    station: str = "",
    *,
    dpi: int = 80,
) -> None:
    residual = y_pred - y_true
    abs_err = np.abs(residual)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_true, y_pred, s=12, alpha=0.55)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set_xlabel("Actual HI (°C)")
    axes[0].set_ylabel("Predicted HI (°C)")
    axes[0].set_title(f"Predicted vs Actual | h{horizon_h} | {station}")
    axes[1].hist(residual, bins=min(40, max(5, len(y_true) // 2)))
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Prediction Error (°C)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Error Distribution | h{horizon_h} | {station}")
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_actual.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    hours = _hours_from_features(X, len(y_true))
    by_hour = pd.DataFrame({"hour": hours, "abs_err": abs_err, "err": residual}).groupby("hour").mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    by_hour["abs_err"].reindex(range(24), fill_value=0).plot(kind="bar", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Absolute Error (°C)")
    ax.set_title(f"Error by Hour | h{horizon_h} | {station}")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_hour.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    stations = _station_names(X, station_labels)
    by_station = pd.DataFrame({"station": stations, "abs_err": abs_err}).groupby("station").mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_station["abs_err"].plot(kind="bar", ax=ax)
    ax.set_xlabel("Station")
    ax.set_ylabel("Mean Absolute Error (°C)")
    ax.set_title(f"Error by Station | h{horizon_h} | {station}")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "error_by_station.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    heat = pd.DataFrame({"station": stations, "hour": hours, "abs_err": abs_err})
    pivot = heat.pivot_table(index="station", columns="hour", values="abs_err", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(3, int(0.8 * len(pivot)))))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Station")
    ax.set_title(f"Error Heatmap | h{horizon_h} | {station}")
    fig.tight_layout()
    fig.savefig(out_dir / "error_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_pi_plots(
    y_true: np.ndarray,
    q05: np.ndarray | None,
    q95: np.ndarray | None,
    out_dir: Path,
    horizon_h: int = 24,
    station: str = "",
    *,
    dpi: int = 80,
) -> None:
    if q05 is None or q95 is None:
        return
    coverage = ((y_true >= q05) & (y_true <= q95)).astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Outside PI", "Inside PI"], [(coverage == 0).mean(), coverage.mean()])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of Predictions")
    ax.set_xlabel("Prediction Interval Coverage")
    ax.set_title(f"PI Calibration | h{horizon_h} | {station}")
    fig.tight_layout()
    fig.savefig(out_dir / "pi_calibration.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(q95 - q05, bins=min(40, max(5, len(y_true) // 2)))
    ax.set_xlabel("Prediction Interval Width (q95 - q05, °C)")
    ax.set_ylabel("Count")
    ax.set_title(f"PI Width Distribution | h{horizon_h} | {station}")
    fig.tight_layout()
    fig.savefig(out_dir / "pi_width.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


class _ReportPDF(FPDF):
    """Custom PDF class with header/footer and helper methods."""

    def __init__(self, station: str, horizon_h: int, backend: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.station = station
        self.horizon_h = horizon_h
        self.backend = backend
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(10, 10, 10)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            6,
            f"HeatShield AI  |  {self.station}  |  h{self.horizon_h}  |  {self.backend}",
            align="R",
        )
        self.ln(6)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def _heading(self, title: str, size: int = 14) -> None:
        self.set_font("Helvetica", "B", size)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def _subheading(self, title: str) -> None:
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(33, 37, 41)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1)

    def _body(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(33, 37, 41)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def _row(self, label: str, value: str, status: str | None = None) -> None:
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(33, 37, 41)
        self.cell(70, 6, label)
        self.set_font("Helvetica", "", 10)
        self.cell(50, 6, value)
        if status:
            self.set_font("Helvetica", "B", 9)
            if status == "ready":
                self.set_text_color(40, 167, 69)
            elif status == "candidate":
                self.set_text_color(255, 140, 0)
            else:
                self.set_text_color(220, 53, 69)
            self.cell(0, 6, f"[{status.upper()}]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.set_text_color(33, 37, 41)
        else:
            self.ln(6)

    def _hr(self) -> None:
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def _image_full_width(self, path: Path, max_h: float = 0) -> None:
        self.image(str(path), x=10, w=190, h=max_h)
        self.ln(3)


def _status_from_metrics(metrics: dict) -> str:
    skill = metrics.get("baselines", {}).get("skill_score", 0.0)
    if skill != skill:
        return "not_ready"
    danger = metrics.get("safety", {}).get("danger_42", {}).get("recall")
    pi = metrics.get("prediction_interval", {})
    coverage = pi.get("coverage_90") if pi.get("available") else None
    if skill < 0.0:
        return "not_ready"
    if coverage is not None and not (0.85 <= coverage <= 0.93):
        return "candidate"
    if skill >= 0.55 and (danger is None or danger != danger or danger >= 0.40):
        return "ready"
    return "candidate"


def _save_feature_importance(
    feature_importance: dict[str, float], out_dir: Path, *, dpi: int = 80
) -> Path | None:
    if not feature_importance:
        return None
    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features = [item[0] for item in reversed(sorted_items)]
    scores = [item[1] for item in reversed(sorted_items)]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importance")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_pdf_report(
    metrics: dict,
    out_path: Path,
    *,
    station: str,
    horizon_h: int,
    backend: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: pd.DataFrame,
    station_labels: Mapping[int, str] | None = None,
    q05: np.ndarray | None = None,
    q95: np.ndarray | None = None,
    hyperparams: dict | None = None,
    feature_importance: dict[str, float] | None = None,
    model_version: str = "v3",
    run_id: str = "",
    eval_dpi: int = 80,
) -> Path:
    """Generate a multi-page PDF evaluation report.

    Returns the path to the written PDF.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    status = _status_from_metrics(metrics)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    reg = metrics.get("regression", {})
    base = metrics.get("baselines", {})
    safety = metrics.get("safety", {})
    pi = metrics.get("prediction_interval", {})
    runtime = metrics.get("runtime", {})
    split_meta = metrics.get("split", {})

    # ------------------------------------------------------------------ #
    # Generate chart images in a temp dir
    # ------------------------------------------------------------------ #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _save_confusion(metrics, tmp, dpi=eval_dpi)
        _save_classification(metrics, tmp, horizon_h=horizon_h, station=station, dpi=eval_dpi)
        _save_error_plots(
            y_true, y_pred, X, tmp, station_labels,
            horizon_h=horizon_h, station=station, dpi=eval_dpi,
        )
        _save_pi_plots(y_true, q05, q95, tmp, horizon_h=horizon_h, station=station, dpi=eval_dpi)
        fi_path = _save_feature_importance(feature_importance, tmp, dpi=eval_dpi)

        # ------------------------------------------------------------------ #
        # Build PDF
        # ------------------------------------------------------------------ #
        pdf = _ReportPDF(station=station, horizon_h=horizon_h, backend=backend)

        # ---- Page 1: Cover ------------------------------------------------
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(33, 37, 41)
        pdf.ln(60)
        pdf.cell(0, 15, "Forecast Evaluation Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, "HeatShield AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(20)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Station:        {station}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.cell(0, 8, f"Horizon:        {horizon_h}h", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.cell(0, 8, f"Backend:        {backend}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.cell(0, 8, f"Model Version:  {model_version}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.cell(0, 8, f"Timestamp:      {ts}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        if run_id:
            pdf.cell(0, 8, f"Run ID:         {run_id}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(15)
        pdf.set_font("Helvetica", "B", 14)
        if status == "ready":
            pdf.set_text_color(40, 167, 69)
        elif status == "candidate":
            pdf.set_text_color(255, 140, 0)
        else:
            pdf.set_text_color(220, 53, 69)
        pdf.cell(0, 10, f"Status: {status.upper()}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        # ---- Page 2: Executive Summary -------------------------------------
        pdf.add_page()
        pdf._heading("Executive Summary")
        pdf._row("MAE", f"{reg.get('mae', float('nan')):.3f} °C")
        pdf._row("RMSE", f"{reg.get('rmse', float('nan')):.3f} °C")
        pdf._row("Skill Score", f"{base.get('skill_score', float('nan')):.3f}", status)
        pdf._row("Bias", f"{reg.get('bias', float('nan')):.3f} °C")
        pdf._row("Correlation", f"{reg.get('correlation', float('nan')):.3f}")
        danger_42 = safety.get("danger_42", {})
        pdf._row("Danger Recall (>=42C)", f"{danger_42.get('recall', float('nan')):.3f}")
        if pi.get("available"):
            pdf._row("PI Coverage 90", f"{pi.get('coverage_90', float('nan')):.3f}")
            pdf._row("Mean PI Width", f"{pi.get('mean_width', float('nan')):.3f} °C")
        else:
            pdf._row("PI Coverage 90", "N/A")
        pdf._hr()
        pdf._subheading("Ship Decision")
        ship = "READY" if status == "ready" else "NOT READY"
        pdf._body(f"Production gate: {ship}")

        # ---- Page 3: Regression Metrics ------------------------------------
        pdf.add_page()
        pdf._heading("Regression Metrics")
        pdf._row("MAE", f"{reg.get('mae', float('nan')):.3f} °C")
        pdf._row("RMSE", f"{reg.get('rmse', float('nan')):.3f} °C")
        pdf._row("Bias", f"{reg.get('bias', float('nan')):.3f} °C")
        pdf._row("Correlation", f"{reg.get('correlation', float('nan')):.3f}")
        pdf._row("P50 Abs Error", f"{reg.get('p50_abs_error', float('nan')):.3f} °C")
        pdf._row("P90 Abs Error", f"{reg.get('p90_abs_error', float('nan')):.3f} °C")
        pdf._row("P95 Abs Error", f"{reg.get('p95_abs_error', float('nan')):.3f} °C")

        # ---- Page 4: Baseline Comparison -----------------------------------
        pdf.add_page()
        pdf._heading("Baseline Comparison")
        pdf._row("Model MAE", f"{reg.get('mae', float('nan')):.3f} °C")
        pdf._row("Persistence MAE", f"{base.get('persistence_mae', float('nan')):.3f} °C")
        pdf._row("Climatology MAE", f"{base.get('climatology_mae', float('nan')):.3f} °C")
        pdf._row("Skill Score", f"{base.get('skill_score', float('nan')):.3f}")
        pdf._hr()
        pdf._body(
            "Skill Score = 1 - MAE_model / max(persistence, climatology). "
            "Values above 0.0 indicate the model beats both baselines."
        )

        # ---- Page 5: Safety & Risk + Confusion Matrix ----------------------
        pdf.add_page()
        pdf._heading("Safety & Risk")
        for key in ["danger_40", "danger_42"]:
            d = safety.get(key, {})
            pdf._subheading(f"Heat Index >= {key.split('_')[1]}°C")
            pdf._row("Support", str(d.get("support", "N/A")))
            pdf._row("Recall", f"{d.get('recall', float('nan')):.3f}")
            pdf._row("False Negative Rate", f"{d.get('false_negative_rate', float('nan')):.3f}")
            pdf.ln(2)
        pdf.add_page()
        pdf._heading("Confusion Matrix")
        pdf._image_full_width(tmp / "confusion_matrix.png")

        # ---- Page 6: Classification Report ---------------------------------
        pdf.add_page()
        pdf._heading("Classification Report")
        pdf._image_full_width(tmp / "classification_report.png")

        # ---- Page 7: Predicted vs Actual -----------------------------------
        pdf.add_page()
        pdf._heading("Predicted vs Actual")
        pdf._image_full_width(tmp / "pred_vs_actual.png")

        # ---- Page 8: Error Analysis ----------------------------------------
        pdf.add_page()
        pdf._heading("Error by Hour")
        pdf._image_full_width(tmp / "error_by_hour.png", max_h=120)
        pdf.add_page()
        pdf._heading("Error by Station")
        pdf._image_full_width(tmp / "error_by_station.png", max_h=120)

        # ---- Page 9: Error Heatmap -----------------------------------------
        pdf.add_page()
        pdf._heading("Error Heatmap")
        pdf._image_full_width(tmp / "error_heatmap.png")

        # ---- Page 10: Prediction Intervals ---------------------------------
        if (tmp / "pi_calibration.png").exists():
            pdf.add_page()
            pdf._heading("Prediction Interval Calibration")
            pdf._image_full_width(tmp / "pi_calibration.png", max_h=110)
        if (tmp / "pi_width.png").exists():
            pdf.add_page()
            pdf._heading("Prediction Interval Width")
            pdf._image_full_width(tmp / "pi_width.png", max_h=110)

        # ---- Page 11: Hyperparameters --------------------------------------
        pdf.add_page()
        pdf._heading("Hyperparameters")
        if hyperparams:
            for k, v in sorted(hyperparams.items()):
                pdf._row(str(k), str(v))
        else:
            pdf._body("No hyperparameters recorded.")

        # ---- Page 12: Feature Importance -----------------------------------
        if fi_path and fi_path.exists():
            pdf.add_page()
            pdf._heading("Feature Importance")
            pdf._image_full_width(fi_path)

        # ---- Page 13: Data Quality -----------------------------------------
        pdf.add_page()
        pdf._heading("Data Quality Summary")
        pdf._row("Feature Rows", str(runtime.get("feature_rows", "N/A")))
        row_counts = split_meta.get("row_counts", {})
        if row_counts:
            pdf._subheading("Split Row Counts")
            for k, v in row_counts.items():
                pdf._row(str(k), str(v))
        gaps = runtime.get("gaps", "N/A")
        outliers = runtime.get("outliers", "N/A")
        pdf._row("Gaps", str(gaps))
        pdf._row("Outliers", str(outliers))
        danger_support = safety.get("danger_42", {}).get("support", 0)
        pdf._row("Danger Support (>=42C)", str(danger_support))

        # ---- Page 14: Runtime & EDR ----------------------------------------
        pdf.add_page()
        pdf._heading("Runtime & EDR Reference")
        pdf._row("Train Time (s)", f"{runtime.get('train_seconds', 'N/A')}")
        pdf._row("Eval Time (s)", f"{runtime.get('eval_seconds', 'N/A')}")
        if run_id:
            pdf._row("Run ID", run_id)
        pdf._row("Generated At", ts)

        pdf.output(str(out_path))
    return out_path
