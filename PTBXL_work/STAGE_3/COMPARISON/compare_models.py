"""
=======================================================================
BLOQUE 1 - ETAPA 3 - COMPARACIÓN GENERAL DE LOS 4 MODELOS
=======================================================================
Lee test_metrics.json de cada modelo ya entrenado y genera una gráfica
comparativa única (AUC/F1 macro + F1 por clase) para la tesis.

Modelos comparados:
  1. CNN1D_ResNet          (STAGE_3/CNN_1D)
  2. CNN-LSTM V1 (último estado oculto)  (STAGE_3/CNN-LSTM)
  3. CNN-LSTM V2 (max pooling temporal)  (STAGE_3/CNN-LSTM-V2)
  4. ECG Transformer V1    (STAGE_3/TRANSFORMER)

Paleta categórica: se usan los 4 primeros slots de la paleta de
referencia del proyecto (azul, naranja, aqua, amarillo) — ese orden
está validado para gráficos de barras (adyacentes), no para scatter/
small-multiples, que exigiría un tope de 3 series.

Salidas (en STAGE_3/COMPARISON/):
  - model_comparison.png   → gráfica de barras agrupadas (2 paneles)
  - model_comparison.csv   → tabla numérica de las mismas métricas
=======================================================================
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ROOT = Path(__file__).resolve().parent.parent   # STAGE_3

# (nombre a mostrar, carpeta de outputs relativa a STAGE_3)
MODELS = [
    ("CNN-ResNet",          "CNN_1D/outputs"),
    ("CNN-LSTM V1",         "CNN-LSTM/outputs"),
    ("CNN-LSTM V2 (max-pool)", "CNN-LSTM-V2/outputs"),
    ("Transformer V1",      "TRANSFORMER/outputs"),
]

# Paleta categórica del proyecto (slots 1-4, validados para barras/adyacentes)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

OUTPUT_DIR = Path(__file__).resolve().parent


def load_all_metrics() -> list[dict]:
    """Carga test_metrics.json de cada modelo. Falla con un mensaje claro
    si a alguno todavía no le corriste el entrenamiento completo."""
    all_metrics = []
    for name, rel_path in MODELS:
        json_path = ROOT / rel_path / "test_metrics.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"No se encontró {json_path}. "
                f"¿Ya corriste el entrenamiento completo de '{name}'?"
            )
        with open(json_path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["_name"] = name
        all_metrics.append(metrics)
    return all_metrics


def plot_comparison(all_metrics: list[dict], path: Path) -> None:
    names = [m["_name"] for m in all_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 5]})

    # ── Panel izquierdo: AUC macro y F1 macro ──────────────────────────
    ax = axes[0]
    summary_keys = ["auc_macro", "f1_macro"]
    summary_labels = ["AUC macro", "F1 macro"]
    n_models = len(all_metrics)
    bar_width = 0.8 / n_models
    x = np.arange(len(summary_keys))

    for i, (m, name, color) in enumerate(zip(all_metrics, names, COLORS)):
        values = [m[k] for k in summary_keys]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Métricas agregadas (test, fold 10)")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Panel derecho: F1 por clase ─────────────────────────────────────
    ax = axes[1]
    x = np.arange(len(SUPERCLASSES))

    for i, (m, name, color) in enumerate(zip(all_metrics, names, COLORS)):
        values = [m[f"f1_{cls}"] for cls in SUPERCLASSES]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(SUPERCLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score por superclase diagnóstica (test, fold 10)")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Comparación de los 4 modelos — PTB-XL multi-label (NORM|MI|STTC|CD|HYP)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison_csv(all_metrics: list[dict], path: Path) -> None:
    import csv
    columns = ["auc_macro", "f1_macro"] + [f"f1_{c}" for c in SUPERCLASSES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + columns)
        for m in all_metrics:
            writer.writerow([m["_name"]] + [round(m[c], 4) for c in columns])


if __name__ == "__main__":
    all_metrics = load_all_metrics()
    plot_comparison(all_metrics, OUTPUT_DIR / "model_comparison.png")
    save_comparison_csv(all_metrics, OUTPUT_DIR / "model_comparison.csv")
    print(f"Comparación guardada en: {OUTPUT_DIR}")
    print("\nResumen:")
    for m in all_metrics:
        print(f"  {m['_name']:<24} AUC={m['auc_macro']:.4f}  F1={m['f1_macro']:.4f}")
