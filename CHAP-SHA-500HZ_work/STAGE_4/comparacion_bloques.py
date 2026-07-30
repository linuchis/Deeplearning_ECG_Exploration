"""
=======================================================================
BLOQUE 2 - ETAPA 4 - COMPARACION VISUAL BLOQUE 1 (PTB-XL) vs BLOQUE 2
                      (Chapman-Shaoxing 500 Hz)
=======================================================================
Bloque 1: PTB-XL, multi-label, 5 superclases (NORM, MI, STTC, CD, HYP).
          Metricas: AUC-ROC macro + F1 macro. Loss: BCEWithLogitsLoss.
Bloque 2: Chapman-Shaoxing 500 Hz, multiclase, 4 clases
          (AFIB, GSVT, SB, SR). Metricas: Accuracy + F1 macro.
          Loss: CrossEntropyLoss. Ventana de 9s (recorte de 1s post-filtro,
          ver STAGE_1_2/exploration_distribution.py).

Los dos bloques resuelven tareas distintas (multi-label vs multiclase)
y con clases distintas, por lo que no existe una metrica 1-a-1 entre
ambos salvo F1 macro. El script deja eso explicito en vez de forzar
una comparacion que no tiene sentido (p.ej. AUC no existe en Bloque 2,
Accuracy no existe en Bloque 1).

Las metricas se LEEN de disco (test_metrics.json de cada modelo, en
ambos bloques) en vez de estar hardcodeadas — asi la comparacion se
actualiza sola si se reentrena cualquier modelo, en cualquiera de los
dos bloques.
  Bloque 1 (PTB-XL, CNN_LSTM/ECGTransformer usan la variante V2, la
  version final de cada arquitectura tras la Etapa 3 del Bloque 1):
    STAGE_3/CNN_1D/outputs/test_metrics.json
    STAGE_3/CNN-LSTM-V2/outputs/test_metrics.json
    STAGE_3/TRANSFORMER-V2/outputs/test_metrics.json
  Bloque 2 (Chapman-Shaoxing):
    STAGE_3/<Modelo>/test_metrics.json

Genera (bilingue, en outputs/es/ y outputs/en/):
  Figura 1 - Ranking de modelos por dataset (F1 macro, barras agrupadas)
  Figura 2 - F1 por clase en cada dataset (2 subplots)
  Figura 3 - Heatmap comparativo de todas las metricas disponibles
  Figura 4 - Bump chart del cambio de ranking entre datasets
  + tabla comparativa (comparacion_bloques.csv, en outputs/, sin idioma)
  + reporte de texto (comparison_report.txt, en es/ y en/)
=======================================================================
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -----------------------------------------------------------------------
# 1. Rutas y carga dinamica de metricas
# -----------------------------------------------------------------------

MODELS = ["CNN1D_ResNet", "CNN_LSTM", "ECGTransformer"]

# Paleta categorica fija del proyecto (no generica) - mismo color para el
# mismo modelo en TODAS las figuras de TODO el proyecto (PTBXL_work y
# CHAP-SHA-500HZ_work comparten esta asignacion), sin importar el dataset.
# Validada con el validador de paletas categoricas (banda de luminosidad,
# piso de croma, separacion bajo daltonismo protan/deutan, piso de vision
# normal).
COLORS = {
    "CNN1D_ResNet":   "darkorchid",
    "CNN_LSTM":       "seagreen",
    "ECGTransformer": "royalblue",
}

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CHAPMAN_CLASSES = ["AFIB", "GSVT", "SB", "SR"]

STAGE4_DIR   = Path(__file__).resolve().parent           # STAGE_4
BLOCK2_DIR   = STAGE4_DIR.parent                          # CHAP-SHA-500HZ_work
PROJECT_ROOT = BLOCK2_DIR.parent                          # Deeplearning_ECG_Exploration
PTBXL_ROOT   = PROJECT_ROOT / "PTBXL_work"

# Bloque 1: CNN_LSTM y ECGTransformer usan la variante V2 (version final
# de cada arquitectura en el Bloque 1 — ver docstring del modulo).
PTBXL_SOURCES = {
    "CNN1D_ResNet":   PTBXL_ROOT / "STAGE_3/CNN_1D/outputs/test_metrics.json",
    "CNN_LSTM":       PTBXL_ROOT / "STAGE_3/CNN-LSTM-V2/outputs/test_metrics.json",
    "ECGTransformer": PTBXL_ROOT / "STAGE_3/TRANSFORMER-V2/outputs/test_metrics.json",
}
CHAPMAN_SOURCES = {
    name: BLOCK2_DIR / "STAGE_3" / "outputs" / name / "test_metrics.json"
    for name in MODELS
}

OUTPUT_DIR    = STAGE4_DIR / "outputs"
OUTPUT_DIR_ES = OUTPUT_DIR / "es"
OUTPUT_DIR_EN = OUTPUT_DIR / "en"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. ¿Ya corriste el entrenamiento/evaluación "
            f"completa de ese modelo?"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all_metrics() -> tuple[dict, dict]:
    """Lee test_metrics.json de los 3 modelos en cada bloque (ver rutas arriba)."""
    ptbxl = {name: _load_json(path) for name, path in PTBXL_SOURCES.items()}
    chapman = {name: _load_json(path) for name, path in CHAPMAN_SOURCES.items()}
    return ptbxl, chapman


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
    """Quita spines sobrantes y deja solo una grilla horizontal tenue."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


TEXTS = {
    "es": {
        "dataset_ptbxl": "PTB-XL", "dataset_chapman": "Chapman-Shaoxing",
        "fig1_title": "Ranking de modelos por dataset (F1 macro)\nPTB-XL (multi-label) vs Chapman-Shaoxing (multiclase)",
        "fig1_ylabel": "F1 macro",
        "fig2_suptitle": "F1-score por clase, por modelo y por dataset",
        "fig2_ylabel": "F1-score",
        "fig3_suptitle": "Heatmap comparativo — todas las métricas disponibles por modelo y dataset",
        "fig3_ptbxl": "PTB-XL (Bloque 1)", "fig3_chapman": "Chapman-Shaoxing (Bloque 2)",
        "fig4_title": "Cambio de ranking entre datasets (según F1 macro)\nBump chart: posición relativa de cada modelo",
        "rank1": "1° (mejor)", "rank2": "2°", "rank3": "3° (peor)",
    },
    "en": {
        "dataset_ptbxl": "PTB-XL", "dataset_chapman": "Chapman-Shaoxing",
        "fig1_title": "Model ranking by dataset (F1 macro)\nPTB-XL (multi-label) vs Chapman-Shaoxing (multiclass)",
        "fig1_ylabel": "F1 macro",
        "fig2_suptitle": "F1-score by class, per model and per dataset",
        "fig2_ylabel": "F1-score",
        "fig3_suptitle": "Comparative heatmap — all available metrics by model and dataset",
        "fig3_ptbxl": "PTB-XL (Block 1)", "fig3_chapman": "Chapman-Shaoxing (Block 2)",
        "fig4_title": "Ranking change between datasets (by F1 macro)\nBump chart: each model's relative position",
        "rank1": "1st (best)", "rank2": "2nd", "rank3": "3rd (worst)",
    },
}


# -----------------------------------------------------------------------
# Figura 1 - Ranking de modelos por dataset (F1 macro)
# -----------------------------------------------------------------------

def figura1_ranking_por_dataset(ptbxl, chapman, path: Path, lang: str):
    t_ = TEXTS[lang]
    dataset_labels = {t_["dataset_ptbxl"]: ptbxl, t_["dataset_chapman"]: chapman}
    datasets = list(dataset_labels.keys())
    x = np.arange(len(datasets))
    n_models = len(MODELS)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, model in enumerate(MODELS):
        values = [dataset_labels[ds][model]["f1_macro"] for ds in datasets]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=model, color=COLORS[model], zorder=3,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(t_["fig1_ylabel"])
    ax.set_title(t_["fig1_title"])
    _clean_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 2 - F1 por clase en cada dataset (2 subplots)
# -----------------------------------------------------------------------

def figura2_f1_por_clase(ptbxl, chapman, path: Path, lang: str):
    t_ = TEXTS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        (t_["dataset_ptbxl"], ptbxl, PTBXL_CLASSES, axes[0]),
        (t_["dataset_chapman"], chapman, CHAPMAN_CLASSES, axes[1]),
    ]

    n_models = len(MODELS)
    bar_width = 0.8 / n_models

    for dataset_name, data, classes, ax in panels:
        x = np.arange(len(classes))
        for i, model in enumerate(MODELS):
            values = [data[model][f"f1_{cls}"] for cls in classes]
            offset = (i - (n_models - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, values, bar_width,
                label=model, color=COLORS[model], zorder=3,
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(t_["fig2_ylabel"])
        ax.set_title(dataset_name)
        _clean_axes(ax)

    axes[0].legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(t_["fig2_suptitle"], fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 3 - Heatmap comparativo de todas las metricas disponibles
# -----------------------------------------------------------------------

def figura3_heatmap(ptbxl, chapman, path: Path, lang: str):
    t_ = TEXTS[lang]
    # Cada dataset tiene sus propias columnas (clases distintas y metrica
    # principal distinta: AUC en Bloque 1, Accuracy en Bloque 2), asi que
    # se muestran como dos heatmaps lado a lado en la misma figura, con
    # la misma escala de color para que sean comparables visualmente.
    ptbxl_cols = ["auc_macro", "f1_macro"] + [f"f1_{c}" for c in PTBXL_CLASSES]
    ptbxl_labels = ["AUC macro", "F1 macro"] + [f"F1 {c}" for c in PTBXL_CLASSES]

    chapman_cols = ["accuracy", "f1_macro"] + [f"f1_{c}" for c in CHAPMAN_CLASSES]
    chapman_labels = ["Accuracy", "F1 macro"] + [f"F1 {c}" for c in CHAPMAN_CLASSES]

    ptbxl_matrix = np.array([[ptbxl[m][c] for c in ptbxl_cols] for m in MODELS])
    chapman_matrix = np.array([[chapman[m][c] for c in chapman_cols] for m in MODELS])

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [len(ptbxl_cols), len(chapman_cols)]},
    )

    common_kwargs = dict(
        vmin=0.35, vmax=1.0, cmap="RdYlGn",
        annot=True, fmt=".3f", cbar=True,
        yticklabels=MODELS, linewidths=1, linecolor="white",
        annot_kws={"fontsize": 8.5},
    )

    sns.heatmap(ptbxl_matrix, ax=axes[0], xticklabels=ptbxl_labels, **common_kwargs)
    axes[0].set_title(t_["fig3_ptbxl"])
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_yticklabels(MODELS, rotation=0, fontsize=9.5)

    sns.heatmap(chapman_matrix, ax=axes[1], xticklabels=chapman_labels, **common_kwargs)
    axes[1].set_title(t_["fig3_chapman"])
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].set_yticklabels(MODELS, rotation=0, fontsize=9.5)

    fig.suptitle(t_["fig3_suptitle"], fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 4 - Bump chart del cambio de ranking (F1 macro)
# -----------------------------------------------------------------------

def figura4_bump_chart(ptbxl, chapman, path: Path, lang: str):
    t_ = TEXTS[lang]
    dataset_labels = {t_["dataset_ptbxl"]: ptbxl, t_["dataset_chapman"]: chapman}
    datasets = list(dataset_labels.keys())

    # rank 1 = mejor F1 macro dentro de cada dataset
    ranks = {}
    for ds in datasets:
        ordered = sorted(MODELS, key=lambda m: dataset_labels[ds][m]["f1_macro"], reverse=True)
        ranks[ds] = {model: ordered.index(model) + 1 for model in MODELS}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = np.arange(len(datasets))

    for model in MODELS:
        y = [ranks[ds][model] for ds in datasets]
        ax.plot(
            x, y, marker="o", markersize=11, linewidth=2.5,
            color=COLORS[model], label=model, zorder=3,
        )
        for xi, yi in zip(x, y):
            f1 = dataset_labels[datasets[xi]][model]["f1_macro"]
            ax.annotate(
                f"{model}\nF1={f1:.3f}",
                xy=(xi, yi),
                xytext=(12 if xi == 0 else -12, 0),
                textcoords="offset points",
                ha="left" if xi == 0 else "right",
                va="center",
                fontsize=8,
                color=COLORS[model],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([t_["rank1"], t_["rank2"], t_["rank3"]])
    ax.invert_yaxis()
    ax.set_ylim(3.5, 0.5)
    ax.set_title(t_["fig4_title"])
    _clean_axes(ax, y_grid=False)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Tabla comparativa (CSV, sin idioma) + reporte de texto (es/en)
# -----------------------------------------------------------------------

def save_comparison_csv(ptbxl, chapman, path: Path) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "auc_macro_or_accuracy", "f1_macro"]
                        + [f"f1_class_{i+1}" for i in range(5)])
        for model in MODELS:
            d = ptbxl[model]
            writer.writerow(["PTB-XL", model, round(d["auc_macro"], 4), round(d["f1_macro"], 4)]
                            + [round(d[f"f1_{c}"], 4) for c in PTBXL_CLASSES])
        for model in MODELS:
            d = chapman[model]
            writer.writerow(["Chapman-Shaoxing", model, round(d["accuracy"], 4), round(d["f1_macro"], 4)]
                            + [round(d[f"f1_{c}"], 4) for c in CHAPMAN_CLASSES] + [""])


def _tabla_texto(ptbxl, chapman, lang: str) -> list[str]:
    es = lang == "es"
    lines = []
    lines.append("=" * 78)
    lines.append("TABLA COMPARATIVA - BLOQUE 1 (PTB-XL) vs BLOQUE 2 (Chapman-Shaoxing)"
                 if es else "COMPARISON TABLE - BLOCK 1 (PTB-XL) vs BLOCK 2 (Chapman-Shaoxing)")
    lines.append("=" * 78)

    lines.append("\n--- Bloque 1: PTB-XL (multi-label, BCEWithLogitsLoss) ---" if es
                else "\n--- Block 1: PTB-XL (multi-label, BCEWithLogitsLoss) ---")
    header = f"{'Modelo' if es else 'Model':<16}{'AUC macro':>11}{'F1 macro':>10}" + \
        "".join(f"{'F1_' + c:>9}" for c in PTBXL_CLASSES)
    lines.append(header)
    lines.append("-" * len(header))
    for model in MODELS:
        d = ptbxl[model]
        lines.append(f"{model:<16}{d['auc_macro']:>11.4f}{d['f1_macro']:>10.4f}" +
                     "".join(f"{d[f'f1_{c}']:>9.4f}" for c in PTBXL_CLASSES))

    lines.append("\n--- Bloque 2: Chapman-Shaoxing 500 Hz, ventana 9s (multiclase, CrossEntropyLoss) ---" if es
                else "\n--- Block 2: Chapman-Shaoxing 500 Hz, 9s window (multiclass, CrossEntropyLoss) ---")
    header = f"{'Modelo' if es else 'Model':<16}{'Accuracy':>11}{'F1 macro':>10}" + \
        "".join(f"{'F1_' + c:>9}" for c in CHAPMAN_CLASSES)
    lines.append(header)
    lines.append("-" * len(header))
    for model in MODELS:
        d = chapman[model]
        lines.append(f"{model:<16}{d['accuracy']:>11.4f}{d['f1_macro']:>10.4f}" +
                     "".join(f"{d[f'f1_{c}']:>9.4f}" for c in CHAPMAN_CLASSES))

    lines.append("\n--- Resumen: F1 macro y ranking en cada dataset ---" if es
                else "\n--- Summary: F1 macro and ranking in each dataset ---")
    header = f"{'Modelo' if es else 'Model':<16}{'F1 PTB-XL':>11}{'Rank':>6}{'F1 Chapman':>13}{'Rank':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    dataset_labels = {"PTB-XL": ptbxl, "Chapman-Shaoxing": chapman}
    ranks = {}
    for ds, data in dataset_labels.items():
        ordered = sorted(MODELS, key=lambda m: data[m]["f1_macro"], reverse=True)
        ranks[ds] = {m: ordered.index(m) + 1 for m in MODELS}
    for model in MODELS:
        f1_p = ptbxl[model]["f1_macro"]
        f1_c = chapman[model]["f1_macro"]
        lines.append(f"{model:<16}{f1_p:>11.4f}{ranks['PTB-XL'][model]:>6}"
                     f"{f1_c:>13.4f}{ranks['Chapman-Shaoxing'][model]:>6}")

    if es:
        lines.append("\nNota: AUC-ROC macro solo existe en Bloque 1 (tarea multi-label);")
        lines.append("Accuracy solo existe en Bloque 2 (tarea multiclase). La unica")
        lines.append("metrica comparable entre ambos bloques es F1 macro.")
        lines.append("Chapman-Shaoxing usa ventanas de 9s (recorte de 1s post-filtro por")
        lines.append("transitorio de borde del filtro — ver STAGE_1_2).")
    else:
        lines.append("\nNote: AUC-ROC macro only exists in Block 1 (multi-label task);")
        lines.append("Accuracy only exists in Block 2 (multiclass task). The only")
        lines.append("metric comparable across both blocks is F1 macro.")
        lines.append("Chapman-Shaoxing uses 9s windows (1s trimmed post-filter due to")
        lines.append("the filter's edge transient — see STAGE_1_2).")
    lines.append("=" * 78)
    return lines


def save_comparison_report(ptbxl, chapman, path: Path, lang: str) -> None:
    path.write_text("\n".join(_tabla_texto(ptbxl, chapman, lang)), encoding="utf-8")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_ES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_EN.mkdir(parents=True, exist_ok=True)

    ptbxl, chapman = load_all_metrics()

    print("\n".join(_tabla_texto(ptbxl, chapman, "es")))

    save_comparison_csv(ptbxl, chapman, OUTPUT_DIR / "comparacion_bloques.csv")
    save_comparison_report(ptbxl, chapman, OUTPUT_DIR_ES / "comparison_report.txt", "es")
    save_comparison_report(ptbxl, chapman, OUTPUT_DIR_EN / "comparison_report.txt", "en")

    for lang, out_dir in [("es", OUTPUT_DIR_ES), ("en", OUTPUT_DIR_EN)]:
        figura1_ranking_por_dataset(ptbxl, chapman, out_dir / "figura1_ranking_por_dataset.png", lang)
        figura2_f1_por_clase(ptbxl, chapman, out_dir / "figura2_f1_por_clase.png", lang)
        figura3_heatmap(ptbxl, chapman, out_dir / "figura3_heatmap_comparativo.png", lang)
        figura4_bump_chart(ptbxl, chapman, out_dir / "figura4_bump_chart_ranking.png", lang)

    print(f"\nTabla: {OUTPUT_DIR / 'comparacion_bloques.csv'}")
    print(f"Figuras + reporte ES en: {OUTPUT_DIR_ES}")
    print(f"Figuras + reporte EN en: {OUTPUT_DIR_EN}")
