"""
=======================================================================
BLOQUE 1 - ETAPA 4 - COMPARACION DE LOS 5 MODELOS (PTB-XL)
=======================================================================
Compara los 5 modelos entrenados en STAGE_3 sobre el mismo dataset y la
misma tarea (PTB-XL, multi-label NORM|MI|STTC|CD|HYP, BCEWithLogitsLoss,
fold 10 = test oficial), en el mismo estilo comparativo que se uso en
CHAP-SHA-500HZ_work/STAGE_4/comparacion_bloques.py: ranking por metrica,
F1 por clase, heatmap y evolucion V1->V2 de las arquitecturas iteradas.

Modelos:
  1. CNN1D_ResNet            (STAGE_3/CNN_1D)
  2. CNN_LSTM V1  (ultimo estado oculto)       (STAGE_3/CNN-LSTM)
  3. CNN_LSTM V2  (max-pool temporal)          (STAGE_3/CNN-LSTM-V2)
  4. ECGTransformer V1                          (STAGE_3/TRANSFORMER)
  5. ECGTransformer V2 (regularizado)           (STAGE_3/TRANSFORMER-V2)

A diferencia de comparacion_bloques.py (que compara 2 datasets distintos
con tareas distintas), aqui los 5 modelos comparten dataset, tarea y
metricas -> SI existe una comparacion 1-a-1 completa (AUC macro, F1
macro, F1 por clase) entre los 5.

Lee test_metrics.json de cada modelo (generado por STAGE_3) en vez de
hardcodear numeros, para que la comparacion se actualice sola si se
reentrena cualquier modelo.

Genera (bilingue, en outputs/es/ y outputs/en/):
  figura1_ranking_metricas        -> AUC macro y F1 macro, barras agrupadas
  figura2_f1_por_clase            -> F1 por superclase, barras agrupadas
  figura3_heatmap_comparativo     -> heatmap de todas las metricas
  figura4_evolucion_v1_v2         -> mejora/retroceso V1 -> V2 (CNN-LSTM y Transformer)
  comparacion_modelos.csv         -> tabla numerica (raiz de outputs/, sin idioma)
  comparison_report.txt (es/en)   -> tabla + conclusiones en texto

Paleta: darkorchid, seagreen, royalblue, darkorange, firebrick — un color
fijo por modelo, consistente en las 4 figuras.
=======================================================================
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ROOT = Path(__file__).resolve().parent.parent   # PTBXL_work

# (clave interna, nombre a mostrar, carpeta de outputs relativa a PTBXL_work)
MODELS = [
    ("CNN1D_ResNet",     "CNN1D_ResNet",              "STAGE_3/CNN_1D/outputs"),
    ("CNN_LSTM_V1",      "CNN_LSTM V1",               "STAGE_3/CNN-LSTM/outputs"),
    ("CNN_LSTM_V2",      "CNN_LSTM V2 (max-pool)",    "STAGE_3/CNN-LSTM-V2/outputs"),
    ("ECGTransformer_V1", "ECGTransformer V1",        "STAGE_3/TRANSFORMER/outputs"),
    ("ECGTransformer_V2", "ECGTransformer V2 (reg.)", "STAGE_3/TRANSFORMER-V2/outputs"),
]
MODEL_KEYS = [key for key, _, _ in MODELS]
MODEL_LABELS = {key: label for key, label, _ in MODELS}

# ── Paleta fija del proyecto (no generica): un color fijo por modelo,
# en el mismo orden en las 4 figuras. Validada con el validador de
# paletas categoricas (banda de luminosidad, piso de croma, separacion
# bajo daltonismo protan/deutan, piso de vision normal).
PALETTE = ["darkorchid", "seagreen", "royalblue", "darkorange", "firebrick"]
MODEL_COLORS = dict(zip(MODEL_KEYS, PALETTE))

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR_ES = OUTPUT_DIR / "es"
OUTPUT_DIR_EN = OUTPUT_DIR / "en"


# -----------------------------------------------------------------------
# 1. Carga de metricas (desde STAGE_3, no hardcodeadas)
# -----------------------------------------------------------------------

def load_all_metrics() -> dict:
    """Carga test_metrics.json de cada modelo ya entrenado en STAGE_3."""
    all_metrics = {}
    for key, _, rel_path in MODELS:
        json_path = ROOT / rel_path / "test_metrics.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"No se encontro {json_path}. "
                f"¿Ya corriste el entrenamiento/evaluacion completa de '{key}'?"
            )
        with open(json_path, encoding="utf-8") as f:
            all_metrics[key] = json.load(f)
    return all_metrics


# -----------------------------------------------------------------------
# 2. Estilo global
# -----------------------------------------------------------------------

sns.set_style("white")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#4a4a4a",
    "axes.labelcolor": "#2b2b2b",
    "text.color": "#2b2b2b",
    "xtick.color": "#2b2b2b",
    "ytick.color": "#2b2b2b",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def _clean_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


TEXTS = {
    "es": {
        "fig1_title": "Ranking de los 5 modelos — PTB-XL (test, fold 10)",
        "fig1_ylabel": "Score",
        "fig2_title": "F1-score por superclase diagnóstica, por modelo (test, fold 10)",
        "fig2_ylabel": "F1-score",
        "fig3_title": "Heatmap comparativo — todas las métricas disponibles",
        "fig4_title": "Evolución V1 → V2: ¿mejoró la iteración de cada arquitectura?",
        "fig4_ylabel": "Score",
        "auc_macro": "AUC macro",
        "f1_macro": "F1 macro",
    },
    "en": {
        "fig1_title": "Ranking of the 5 models — PTB-XL (test, fold 10)",
        "fig1_ylabel": "Score",
        "fig2_title": "F1-score by diagnostic superclass, per model (test, fold 10)",
        "fig2_ylabel": "F1-score",
        "fig3_title": "Comparative heatmap — all available metrics",
        "fig4_title": "V1 → V2 evolution: did each architecture's iteration improve?",
        "fig4_ylabel": "Score",
        "auc_macro": "AUC macro",
        "f1_macro": "F1 macro",
    },
}


# -----------------------------------------------------------------------
# Figura 1 - Ranking (AUC macro y F1 macro) de los 5 modelos
# -----------------------------------------------------------------------

def figura1_ranking(all_metrics: dict, path: Path, lang: str):
    t = TEXTS[lang]
    summary_keys = ["auc_macro", "f1_macro"]
    summary_labels = [t["auc_macro"], t["f1_macro"]]

    x = np.arange(len(summary_keys))
    n_models = len(MODEL_KEYS)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, key in enumerate(MODEL_KEYS):
        values = [all_metrics[key][k] for k in summary_keys]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=MODEL_LABELS[key], color=MODEL_COLORS[key], zorder=3,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_labels, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(t["fig1_ylabel"])
    ax.set_title(t["fig1_title"])
    _clean_axes(ax)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=1)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 2 - F1 por clase, por modelo
# -----------------------------------------------------------------------

def figura2_f1_por_clase(all_metrics: dict, path: Path, lang: str):
    t = TEXTS[lang]
    x = np.arange(len(SUPERCLASSES))
    n_models = len(MODEL_KEYS)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, key in enumerate(MODEL_KEYS):
        values = [all_metrics[key][f"f1_{cls}"] for cls in SUPERCLASSES]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=MODEL_LABELS[key], color=MODEL_COLORS[key], zorder=3,
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(SUPERCLASSES, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(t["fig2_ylabel"])
    ax.set_title(t["fig2_title"])
    _clean_axes(ax)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 3 - Heatmap comparativo de todas las metricas
# -----------------------------------------------------------------------

def figura3_heatmap(all_metrics: dict, path: Path, lang: str):
    t = TEXTS[lang]
    cols = ["auc_macro", "f1_macro"] + [f"f1_{c}" for c in SUPERCLASSES]
    col_labels = [t["auc_macro"], t["f1_macro"]] + [f"F1 {c}" for c in SUPERCLASSES]
    labels = [MODEL_LABELS[k] for k in MODEL_KEYS]

    matrix = np.array([[all_metrics[k][c] for c in cols] for k in MODEL_KEYS])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(
        matrix, ax=ax, xticklabels=col_labels, yticklabels=labels,
        vmin=0.35, vmax=1.0, cmap="RdYlGn",
        annot=True, fmt=".3f", cbar=True,
        linewidths=1, linecolor="white",
        annot_kws={"fontsize": 8.5},
    )
    ax.set_title(t["fig3_title"])
    ax.tick_params(axis="x", rotation=35)
    ax.set_yticklabels(labels, rotation=0, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 4 - Evolucion V1 -> V2 (CNN_LSTM y ECGTransformer)
# -----------------------------------------------------------------------

def figura4_evolucion_v1_v2(all_metrics: dict, path: Path, lang: str):
    t = TEXTS[lang]
    pairs = [
        ("CNN_LSTM", "CNN_LSTM_V1", "CNN_LSTM_V2"),
        ("ECGTransformer", "ECGTransformer_V1", "ECGTransformer_V2"),
    ]
    metrics = [("auc_macro", t["auc_macro"]), ("f1_macro", t["f1_macro"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        x = np.arange(len(pairs))
        width = 0.32
        for i, (arch_name, key_v1, key_v2) in enumerate(pairs):
            v1 = all_metrics[key_v1][metric_key]
            v2 = all_metrics[key_v2][metric_key]
            ax.bar(i - width / 2, v1, width, color=MODEL_COLORS[key_v1], zorder=3)
            ax.bar(i + width / 2, v2, width, color=MODEL_COLORS[key_v2], zorder=3)
            ax.text(i - width / 2, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=8)
            ax.text(i + width / 2, v2 + 0.01, f"{v2:.3f}", ha="center", fontsize=8)
            delta = v2 - v1
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            ax.text(i, max(v1, v2) + 0.06, f"{arrow} {delta:+.3f}",
                    ha="center", fontsize=9, fontweight="bold",
                    color="#1a7a1a" if delta > 0 else "#b03030")

        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in pairs], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(t["fig4_ylabel"])
        ax.set_title(metric_label)
        _clean_axes(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS["CNN_LSTM_V1"], label="V1"),
        plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS["CNN_LSTM_V2"], label="V2"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)

    fig.suptitle(t["fig4_title"], fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Tabla numerica (CSV, sin idioma) + reporte de texto (es/en)
# -----------------------------------------------------------------------

def save_comparison_csv(all_metrics: dict, path: Path) -> None:
    import csv
    columns = ["auc_macro", "f1_macro"] + [f"f1_{c}" for c in SUPERCLASSES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + columns)
        for key in MODEL_KEYS:
            m = all_metrics[key]
            writer.writerow([MODEL_LABELS[key]] + [round(m[c], 4) for c in columns])


def _ranking_lines(all_metrics: dict, metric_key: str) -> list[str]:
    ordered = sorted(MODEL_KEYS, key=lambda k: all_metrics[k][metric_key], reverse=True)
    return [f"  {i+1}. {MODEL_LABELS[k]:<28} {all_metrics[k][metric_key]:.4f}"
            for i, k in enumerate(ordered)]


def save_comparison_report_es(all_metrics: dict, path: Path) -> None:
    lines = [
        "=" * 78,
        "COMPARACION DE LOS 5 MODELOS - PTB-XL (test, fold 10)",
        "=" * 78, "",
        "Tabla de metricas:",
        f"{'Modelo':<28}{'AUC macro':>11}{'F1 macro':>10}" +
        "".join(f"{'F1_' + c:>9}" for c in SUPERCLASSES),
    ]
    lines.append("-" * len(lines[-1]))
    for key in MODEL_KEYS:
        m = all_metrics[key]
        lines.append(
            f"{MODEL_LABELS[key]:<28}{m['auc_macro']:>11.4f}{m['f1_macro']:>10.4f}" +
            "".join(f"{m[f'f1_{c}']:>9.4f}" for c in SUPERCLASSES)
        )
    lines += ["", "Ranking por AUC macro:"] + _ranking_lines(all_metrics, "auc_macro")
    lines += ["", "Ranking por F1 macro:"] + _ranking_lines(all_metrics, "f1_macro")

    lines += ["", "Evolucion V1 -> V2:"]
    for arch, k1, k2 in [("CNN_LSTM", "CNN_LSTM_V1", "CNN_LSTM_V2"),
                         ("ECGTransformer", "ECGTransformer_V1", "ECGTransformer_V2")]:
        d_auc = all_metrics[k2]["auc_macro"] - all_metrics[k1]["auc_macro"]
        d_f1 = all_metrics[k2]["f1_macro"] - all_metrics[k1]["f1_macro"]
        lines.append(f"  {arch:<16} AUC: {d_auc:+.4f}   F1 macro: {d_f1:+.4f}")

    lines.append("")
    lines.append("Nota: los 5 modelos comparten dataset, tarea y metricas (a diferencia")
    lines.append("de la comparacion entre bloques de CHAP-SHA-500HZ_work), por lo que")
    lines.append("todas las metricas son directamente comparables 1 a 1.")
    lines.append("=" * 78)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_comparison_report_en(all_metrics: dict, path: Path) -> None:
    lines = [
        "=" * 78,
        "COMPARISON OF THE 5 MODELS - PTB-XL (test, fold 10)",
        "=" * 78, "",
        "Metrics table:",
        f"{'Model':<28}{'AUC macro':>11}{'F1 macro':>10}" +
        "".join(f"{'F1_' + c:>9}" for c in SUPERCLASSES),
    ]
    lines.append("-" * len(lines[-1]))
    for key in MODEL_KEYS:
        m = all_metrics[key]
        lines.append(
            f"{MODEL_LABELS[key]:<28}{m['auc_macro']:>11.4f}{m['f1_macro']:>10.4f}" +
            "".join(f"{m[f'f1_{c}']:>9.4f}" for c in SUPERCLASSES)
        )
    lines += ["", "Ranking by AUC macro:"] + _ranking_lines(all_metrics, "auc_macro")
    lines += ["", "Ranking by F1 macro:"] + _ranking_lines(all_metrics, "f1_macro")

    lines += ["", "V1 -> V2 evolution:"]
    for arch, k1, k2 in [("CNN_LSTM", "CNN_LSTM_V1", "CNN_LSTM_V2"),
                         ("ECGTransformer", "ECGTransformer_V1", "ECGTransformer_V2")]:
        d_auc = all_metrics[k2]["auc_macro"] - all_metrics[k1]["auc_macro"]
        d_f1 = all_metrics[k2]["f1_macro"] - all_metrics[k1]["f1_macro"]
        lines.append(f"  {arch:<16} AUC: {d_auc:+.4f}   F1 macro: {d_f1:+.4f}")

    lines.append("")
    lines.append("Note: all 5 models share the same dataset, task and metrics (unlike")
    lines.append("the cross-dataset comparison in CHAP-SHA-500HZ_work), so every metric")
    lines.append("is directly comparable 1-to-1 across models.")
    lines.append("=" * 78)
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_ES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_EN.mkdir(parents=True, exist_ok=True)

    all_metrics = load_all_metrics()

    save_comparison_csv(all_metrics, OUTPUT_DIR / "comparacion_modelos.csv")
    save_comparison_report_es(all_metrics, OUTPUT_DIR_ES / "comparison_report.txt")
    save_comparison_report_en(all_metrics, OUTPUT_DIR_EN / "comparison_report.txt")

    for lang, out_dir in [("es", OUTPUT_DIR_ES), ("en", OUTPUT_DIR_EN)]:
        figura1_ranking(all_metrics, out_dir / "figura1_ranking_metricas.png", lang)
        figura2_f1_por_clase(all_metrics, out_dir / "figura2_f1_por_clase.png", lang)
        figura3_heatmap(all_metrics, out_dir / "figura3_heatmap_comparativo.png", lang)
        figura4_evolucion_v1_v2(all_metrics, out_dir / "figura4_evolucion_v1_v2.png", lang)

    print(f"Tabla numerica: {OUTPUT_DIR / 'comparacion_modelos.csv'}")
    print(f"Graficos + reporte ES en: {OUTPUT_DIR_ES}")
    print(f"Graficos + reporte EN en: {OUTPUT_DIR_EN}")
    print("\nResumen (F1 macro, de mayor a menor):")
    for key in sorted(MODEL_KEYS, key=lambda k: all_metrics[k]["f1_macro"], reverse=True):
        m = all_metrics[key]
        print(f"  {MODEL_LABELS[key]:<28} AUC={m['auc_macro']:.4f}  F1={m['f1_macro']:.4f}")
