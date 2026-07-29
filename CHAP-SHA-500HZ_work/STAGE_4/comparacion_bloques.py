"""
=======================================================================
BLOQUE 2 - ETAPA 4 - COMPARACION VISUAL BLOQUE 1 (PTB-XL) vs BLOQUE 2
                      (Chapman-Shaoxing 500 Hz)
=======================================================================
Bloque 1: PTB-XL, multi-label, 5 superclases (NORM, MI, STTC, CD, HYP).
          Metricas: AUC-ROC macro + F1 macro. Loss: BCEWithLogitsLoss.
Bloque 2: Chapman-Shaoxing 500 Hz, multiclase, 4 clases
          (AFIB, GSVT, SB, SR). Metricas: Accuracy + F1 macro.
          Loss: CrossEntropyLoss.

Los dos bloques resuelven tareas distintas (multi-label vs multiclase)
y con clases distintas, por lo que no existe una metrica 1-a-1 entre
ambos salvo F1 macro. El script deja eso explicito en vez de forzar
una comparacion que no tiene sentido (p.ej. AUC no existe en Bloque 2,
Accuracy no existe en Bloque 1).

Genera:
  Figura 1 - Ranking de modelos por dataset (F1 macro, barras agrupadas)
  Figura 2 - F1 por clase en cada dataset (2 subplots)
  Figura 3 - Heatmap comparativo de todas las metricas disponibles
  Figura 4 - Bump chart del cambio de ranking entre datasets
  + tabla comparativa impresa en consola

Todos los datos estan hardcodeados (resultados ya obtenidos en las
Etapas anteriores de ambos bloques) - el script no lee archivos
externos y puede ejecutarse de forma aislada.
=======================================================================
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -----------------------------------------------------------------------
# 1. Datos hardcodeados
# -----------------------------------------------------------------------

MODELS = ["CNN1D_ResNet", "CNN_LSTM", "ECGTransformer"]

# Paleta categorica fija del proyecto - mismo color para el mismo modelo
# en TODAS las figuras, sin importar el dataset.
COLORS = {
    "CNN1D_ResNet": "#E63946",
    "CNN_LSTM": "#F4A261",
    "ECGTransformer": "#2A9D8F",
}

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CHAPMAN_CLASSES = ["AFIB", "GSVT", "SB", "SR"]

# Bloque 1 - PTB-XL (multi-label, BCEWithLogitsLoss)
# CNN_LSTM corresponde a la version V2 (max-pool temporal) y
# ECGTransformer a la version V2 - las variantes finales de cada
# arquitectura tras la Etapa 3 del Bloque 1.
PTBXL = {
    "CNN1D_ResNet": {
        "auc_macro": 0.9185, "f1_macro": 0.7129,
        "f1_NORM": 0.8774, "f1_MI": 0.7153, "f1_STTC": 0.7718,
        "f1_CD": 0.7509, "f1_HYP": 0.4492,
    },
    "CNN_LSTM": {
        "auc_macro": 0.9141, "f1_macro": 0.6979,
        "f1_NORM": 0.8784, "f1_MI": 0.7226, "f1_STTC": 0.7359,
        "f1_CD": 0.7514, "f1_HYP": 0.4010,
    },
    "ECGTransformer": {
        "auc_macro": 0.8956, "f1_macro": 0.6643,
        "f1_NORM": 0.8622, "f1_MI": 0.6135, "f1_STTC": 0.7284,
        "f1_CD": 0.7151, "f1_HYP": 0.4022,
    },
}

# Bloque 2 - Chapman-Shaoxing 500 Hz (multiclase, CrossEntropyLoss)
CHAPMAN = {
    "CNN1D_ResNet": {
        "accuracy": 0.9580, "f1_macro": 0.9537,
        "f1_AFIB": 0.9464, "f1_GSVT": 0.9355, "f1_SB": 0.9809, "f1_SR": 0.9521,
    },
    "CNN_LSTM": {
        "accuracy": 0.9542, "f1_macro": 0.9506,
        "f1_AFIB": 0.9525, "f1_GSVT": 0.9336, "f1_SB": 0.9742, "f1_SR": 0.9422,
    },
    "ECGTransformer": {
        "accuracy": 0.9320, "f1_macro": 0.9275,
        "f1_AFIB": 0.9381, "f1_GSVT": 0.9126, "f1_SB": 0.9561, "f1_SR": 0.9034,
    },
}

DATASET_LABELS = {"PTB-XL": PTBXL, "Chapman-Shaoxing": CHAPMAN}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "etapa4_comparativa"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


# -----------------------------------------------------------------------
# Figura 1 - Ranking de modelos por dataset (F1 macro)
# -----------------------------------------------------------------------

def figura1_ranking_por_dataset():
    datasets = list(DATASET_LABELS.keys())
    x = np.arange(len(datasets))
    n_models = len(MODELS)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, model in enumerate(MODELS):
        values = [DATASET_LABELS[ds][model]["f1_macro"] for ds in datasets]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=model, color=COLORS[model], zorder=3,
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 macro")
    ax.set_title(
        "Ranking de modelos por dataset (F1 macro)\n"
        "PTB-XL (multi-label) vs Chapman-Shaoxing (multiclase)"
    )
    _clean_axes(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figura1_ranking_por_dataset.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 2 - F1 por clase en cada dataset (2 subplots)
# -----------------------------------------------------------------------

def figura2_f1_por_clase():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        ("PTB-XL", PTBXL, PTBXL_CLASSES, axes[0]),
        ("Chapman-Shaoxing", CHAPMAN, CHAPMAN_CLASSES, axes[1]),
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
        ax.set_ylabel("F1-score")
        ax.set_title(f"{dataset_name}")
        _clean_axes(ax)

    axes[0].legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle("F1-score por clase, por modelo y por dataset", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUTPUT_DIR / "figura2_f1_por_clase.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 3 - Heatmap comparativo de todas las metricas disponibles
# -----------------------------------------------------------------------

def figura3_heatmap():
    # Cada dataset tiene sus propias columnas (clases distintas y metrica
    # principal distinta: AUC en Bloque 1, Accuracy en Bloque 2), asi que
    # se muestran como dos heatmaps lado a lado en la misma figura, con
    # la misma escala de color para que sean comparables visualmente.
    ptbxl_cols = ["auc_macro", "f1_macro"] + [f"f1_{c}" for c in PTBXL_CLASSES]
    ptbxl_labels = ["AUC macro", "F1 macro"] + [f"F1 {c}" for c in PTBXL_CLASSES]

    chapman_cols = ["accuracy", "f1_macro"] + [f"f1_{c}" for c in CHAPMAN_CLASSES]
    chapman_labels = ["Accuracy", "F1 macro"] + [f"F1 {c}" for c in CHAPMAN_CLASSES]

    ptbxl_matrix = np.array([[PTBXL[m][c] for c in ptbxl_cols] for m in MODELS])
    chapman_matrix = np.array([[CHAPMAN[m][c] for c in chapman_cols] for m in MODELS])

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
    axes[0].set_title("PTB-XL (Bloque 1)")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_yticklabels(MODELS, rotation=0, fontsize=9.5)

    sns.heatmap(chapman_matrix, ax=axes[1], xticklabels=chapman_labels, **common_kwargs)
    axes[1].set_title("Chapman-Shaoxing (Bloque 2)")
    axes[1].tick_params(axis="x", rotation=40)
    axes[1].set_yticklabels(MODELS, rotation=0, fontsize=9.5)

    fig.suptitle(
        "Heatmap comparativo — todas las metricas disponibles por modelo y dataset",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "figura3_heatmap_comparativo.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Figura 4 - Bump chart del cambio de ranking (F1 macro)
# -----------------------------------------------------------------------

def figura4_bump_chart():
    datasets = list(DATASET_LABELS.keys())

    # rank 1 = mejor F1 macro dentro de cada dataset
    ranks = {}
    for ds in datasets:
        ordered = sorted(MODELS, key=lambda m: DATASET_LABELS[ds][m]["f1_macro"], reverse=True)
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
            f1 = DATASET_LABELS[datasets[xi]][model]["f1_macro"]
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
    ax.set_yticklabels(["1° (mejor)", "2°", "3° (peor)"])
    ax.invert_yaxis()
    ax.set_ylim(3.5, 0.5)
    ax.set_title(
        "Cambio de ranking entre datasets (segun F1 macro)\n"
        "Bump chart: posicion relativa de cada modelo"
    )
    _clean_axes(ax, y_grid=False)
    ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figura4_bump_chart_ranking.png", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Tabla comparativa en consola
# -----------------------------------------------------------------------

def imprimir_tabla_comparativa():
    print("=" * 78)
    print("TABLA COMPARATIVA - BLOQUE 1 (PTB-XL) vs BLOQUE 2 (Chapman-Shaoxing)")
    print("=" * 78)

    print("\n--- Bloque 1: PTB-XL (multi-label, BCEWithLogitsLoss) ---")
    header = f"{'Modelo':<16}{'AUC macro':>11}{'F1 macro':>10}" + \
        "".join(f"{'F1_' + c:>9}" for c in PTBXL_CLASSES)
    print(header)
    print("-" * len(header))
    for model in MODELS:
        d = PTBXL[model]
        row = f"{model:<16}{d['auc_macro']:>11.4f}{d['f1_macro']:>10.4f}" + \
            "".join(f"{d[f'f1_{c}']:>9.4f}" for c in PTBXL_CLASSES)
        print(row)

    print("\n--- Bloque 2: Chapman-Shaoxing 500 Hz (multiclase, CrossEntropyLoss) ---")
    header = f"{'Modelo':<16}{'Accuracy':>11}{'F1 macro':>10}" + \
        "".join(f"{'F1_' + c:>9}" for c in CHAPMAN_CLASSES)
    print(header)
    print("-" * len(header))
    for model in MODELS:
        d = CHAPMAN[model]
        row = f"{model:<16}{d['accuracy']:>11.4f}{d['f1_macro']:>10.4f}" + \
            "".join(f"{d[f'f1_{c}']:>9.4f}" for c in CHAPMAN_CLASSES)
        print(row)

    print("\n--- Resumen: F1 macro y ranking en cada dataset ---")
    header = f"{'Modelo':<16}{'F1 PTB-XL':>11}{'Rank':>6}{'F1 Chapman':>13}{'Rank':>6}"
    print(header)
    print("-" * len(header))
    ranks = {}
    for ds, data in DATASET_LABELS.items():
        ordered = sorted(MODELS, key=lambda m: data[m]["f1_macro"], reverse=True)
        ranks[ds] = {m: ordered.index(m) + 1 for m in MODELS}
    for model in MODELS:
        f1_p = PTBXL[model]["f1_macro"]
        f1_c = CHAPMAN[model]["f1_macro"]
        print(f"{model:<16}{f1_p:>11.4f}{ranks['PTB-XL'][model]:>6}"
              f"{f1_c:>13.4f}{ranks['Chapman-Shaoxing'][model]:>6}")

    print("\nNota: AUC-ROC macro solo existe en Bloque 1 (tarea multi-label);")
    print("Accuracy solo existe en Bloque 2 (tarea multiclase). La unica")
    print("metrica comparable entre ambos bloques es F1 macro.")
    print("=" * 78)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    imprimir_tabla_comparativa()

    figura1_ranking_por_dataset()
    figura2_f1_por_clase()
    figura3_heatmap()
    figura4_bump_chart()

    print(f"\nFiguras guardadas en: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {f.name}")
