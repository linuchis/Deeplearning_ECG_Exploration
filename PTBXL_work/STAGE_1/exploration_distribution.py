"""
STAGE 1 - Exploracion de distribucion de clases (PTB-XL)

Objetivo:
    Cuantificar el desbalance de clases del dataset PTB-XL a nivel de
    superclase diagnostica (NORM, MI, STTC, CD, HYP) y dejar un mapa de
    distribucion (tablas + graficos) que sirva de base para decidir la
    estrategia de correccion antes de entrenar.

    La agregacion scp_codes -> diagnostic_superclass replica exactamente
    el procedimiento oficial descrito en example_physionet.py (PhysioNet)
    y en el paper de PTB-XL (Wagner et al., 2020), para que el analisis
    sea comparable con lo ya publicado.

Salidas (en STAGE_1/outputs/):
    - class_distribution_superclass.csv   -> conteo/porcentaje por clase (multi-label)
    - class_distribution_by_fold.csv      -> conteo por clase y strat_fold (1-10)
    - label_cardinality.csv               -> distribucion del numero de etiquetas por registro
    - class_weights_recommendation.csv    -> pesos sugeridos para weighted cross-entropy
    - class_distribution_overview.png     -> grafico de barras del conteo por clase
    - class_distribution_by_fold.png      -> heatmap de conteo por clase x fold
    - class_cooccurrence.png              -> heatmap de co-ocurrencia entre clases
    - summary_report.txt                  -> resumen textual con el diagnostico de desbalance

Uso:
    python exploration_distribution.py --data-dir "C:\\Users\\Lina Castañeda\\Documents\\Originals\\PTBXL-RAW"
"""

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_args():
    parser = argparse.ArgumentParser(description="Exploracion de distribucion de clases PTB-XL")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\Lina Castañeda\Documents\Originals\PTBXL-RAW",
        help="Carpeta raiz del dataset PTB-XL (contiene ptbxl_database.csv y scp_statements.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "outputs"),
        help="Carpeta donde se guardan tablas y graficos",
    )
    return parser.parse_args()


def load_metadata(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def build_superclass_labels(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Agrega scp_codes -> diagnostic_superclass (NORM/MI/STTC/CD/HYP), igual que example_physionet.py."""
    agg_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(scp_dict):
        classes = set()
        for code in scp_dict.keys():
            if code in agg_df.index:
                classes.add(agg_df.loc[code].diagnostic_class)
        return sorted(classes)

    df = df.copy()
    df["diagnostic_superclass"] = df["scp_codes"].apply(aggregate_diagnostic)
    df["n_labels"] = df["diagnostic_superclass"].apply(len)
    return df


def multilabel_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz binaria ecg_id x superclase (un registro puede tener varias etiquetas o ninguna)."""
    mat = pd.DataFrame(0, index=df.index, columns=SUPERCLASSES)
    for ecg_id, labels in df["diagnostic_superclass"].items():
        for label in labels:
            mat.loc[ecg_id, label] = 1
    return mat


def compute_class_distribution(mat: pd.DataFrame, n_total: int) -> pd.DataFrame:
    counts = mat.sum(axis=0).astype(int)
    dist = pd.DataFrame({
        "class": counts.index,
        "n_records": counts.values,
        "pct_of_total_records": (counts.values / n_total * 100).round(2),
    })
    n_unlabeled = int((mat.sum(axis=1) == 0).sum())
    dist = pd.concat([dist, pd.DataFrame([{
        "class": "NO_DIAGNOSTIC_LABEL",
        "n_records": n_unlabeled,
        "pct_of_total_records": round(n_unlabeled / n_total * 100, 2),
    }])], ignore_index=True)
    dist = dist.sort_values("n_records", ascending=False).reset_index(drop=True)
    return dist


def compute_imbalance_ratio(dist: pd.DataFrame) -> tuple[float, str, str]:
    labeled = dist[dist["class"] != "NO_DIAGNOSTIC_LABEL"]
    max_row = labeled.loc[labeled["n_records"].idxmax()]
    min_row = labeled.loc[labeled["n_records"].idxmin()]
    ratio = max_row["n_records"] / min_row["n_records"]
    return ratio, max_row["class"], min_row["class"]


def compute_class_weights(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Pesos para weighted cross-entropy (esquema multi-label / BCEWithLogitsLoss):
        pos_weight_c = n_negatives_c / n_positives_c
    y tambien el esquema clasico "inverse frequency" normalizado
    (n_total / (n_classes * n_positives_c)), equivalente al class_weight='balanced' de sklearn.
    """
    n_total = len(mat)
    n_classes = len(SUPERCLASSES)
    rows = []
    for c in SUPERCLASSES:
        n_pos = int(mat[c].sum())
        n_neg = n_total - n_pos
        pos_weight = n_neg / n_pos if n_pos > 0 else float("nan")
        balanced_weight = n_total / (n_classes * n_pos) if n_pos > 0 else float("nan")
        rows.append({
            "class": c,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "pos_weight_bce": round(pos_weight, 4),
            "balanced_weight_sklearn_style": round(balanced_weight, 4),
        })
    return pd.DataFrame(rows)


def plot_class_distribution(dist: pd.DataFrame, output_path: Path):
    plot_df = dist[dist["class"] != "NO_DIAGNOSTIC_LABEL"].sort_values("n_records", ascending=False)
    unlabeled = dist[dist["class"] == "NO_DIAGNOSTIC_LABEL"]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_df)))
    bars = ax.bar(plot_df["class"], plot_df["n_records"], color=colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("Distribucion de clases - PTB-XL (superclase diagnostica)")
    ax.set_xlabel("Superclase diagnostica")
    ax.set_ylabel("Numero de registros (multi-label)")
    if not unlabeled.empty:
        ax.text(
            0.99, 0.97,
            f"Sin etiqueta diagnostica: {int(unlabeled['n_records'].iloc[0])} "
            f"({unlabeled['pct_of_total_records'].iloc[0]}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="gray",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_distribution_by_fold(df: pd.DataFrame, mat: pd.DataFrame, output_path: Path):
    mat_fold = mat.copy()
    mat_fold["strat_fold"] = df["strat_fold"]
    by_fold = mat_fold.groupby("strat_fold")[SUPERCLASSES].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(by_fold.T.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(by_fold.index)))
    ax.set_xticklabels(by_fold.index)
    ax.set_yticks(range(len(SUPERCLASSES)))
    ax.set_yticklabels(SUPERCLASSES)
    ax.set_xlabel("strat_fold (10 = fold de test recomendado por PTB-XL)")
    ax.set_title("Conteo por clase y fold estratificado")
    for i in range(len(SUPERCLASSES)):
        for j in range(len(by_fold.index)):
            ax.text(j, i, int(by_fold.T.values[i, j]), ha="center", va="center",
                     color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="n_records")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return by_fold


def plot_cooccurrence(mat: pd.DataFrame, output_path: Path):
    cooc = mat.T.dot(mat)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cooc.values, cmap="magma")
    ax.set_xticks(range(len(SUPERCLASSES)))
    ax.set_xticklabels(SUPERCLASSES)
    ax.set_yticks(range(len(SUPERCLASSES)))
    ax.set_yticklabels(SUPERCLASSES)
    ax.set_title("Co-ocurrencia entre superclases (diagonal = total por clase)")
    for i in range(len(SUPERCLASSES)):
        for j in range(len(SUPERCLASSES)):
            ax.text(j, i, int(cooc.values[i, j]), ha="center", va="center",
                     color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return cooc


def write_summary(
    output_dir: Path,
    n_total: int,
    dist: pd.DataFrame,
    ratio: float,
    max_class: str,
    min_class: str,
    weights: pd.DataFrame,
    cardinality: pd.Series,
):
    lines = []
    lines.append("=" * 70)
    lines.append("PTB-XL - Mapa de distribucion de clases (STAGE 1)")
    lines.append("=" * 70)
    lines.append(f"Total de registros (ECGs): {n_total}")
    lines.append("")
    lines.append("Distribucion por superclase diagnostica (multi-label):")
    for _, row in dist.iterrows():
        lines.append(f"  - {row['class']:<22} {row['n_records']:>6} registros  ({row['pct_of_total_records']}%)")
    lines.append("")
    lines.append(f"Razon de desbalance (clase mayoritaria / clase minoritaria): {ratio:.2f}x")
    lines.append(f"  Mayoritaria: {max_class}  |  Minoritaria: {min_class}")
    lines.append("")

    if ratio >= 5:
        veredicto = "DESBALANCE SEVERO"
    elif ratio >= 2:
        veredicto = "DESBALANCE MODERADO"
    else:
        veredicto = "DESBALANCE LEVE"
    lines.append(f"Diagnostico: {veredicto} (ratio {ratio:.2f}x)")
    lines.append("")

    lines.append("Cardinalidad de etiquetas por registro (cuantas superclases por ECG):")
    for n_labels, count in cardinality.items():
        lines.append(f"  - {n_labels} etiqueta(s): {count} registros")
    lines.append("")

    lines.append("Estrategia recomendada: Weighted Cross-Entropy")
    lines.append("  (consistente con la literatura de PTB-XL, p.ej. Wagner et al. 2020 "
                  "y trabajos posteriores de clasificacion multi-label de ECG).")
    lines.append("  Pesos sugeridos por clase (ver class_weights_recommendation.csv):")
    for _, row in weights.iterrows():
        lines.append(
            f"    - {row['class']:<6} pos_weight(BCE)={row['pos_weight_bce']:>8} "
            f"| balanced_weight={row['balanced_weight_sklearn_style']:>8}"
        )
    lines.append("")
    lines.append("Notas de aplicacion:")
    lines.append("  - PTB-XL es multi-label (un ECG puede tener varias superclases o ninguna).")
    lines.append("  - Para BCEWithLogitsLoss(pos_weight=...) usar la columna 'pos_weight_bce'.")
    lines.append("  - Para CrossEntropyLoss en un escenario single-label (una superclase por ECG,")
    lines.append("    tras resolver casos multi-etiqueta), usar 'balanced_weight_sklearn_style'.")
    lines.append("  - strat_fold ya viene balanceado por el dataset (10 folds, fold 10 = test oficial);")
    lines.append("    verificar en class_distribution_by_fold.csv que el desbalance se mantiene similar")
    lines.append("    en todos los folds antes de entrenar.")
    lines.append("=" * 70)

    (output_dir / "summary_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando metadata desde: {data_dir}")
    df = load_metadata(data_dir)
    df = build_superclass_labels(df, data_dir)
    n_total = len(df)

    mat = multilabel_matrix(df)

    dist = compute_class_distribution(mat, n_total)
    dist.to_csv(output_dir / "class_distribution_superclass.csv", index=False)

    ratio, max_class, min_class = compute_imbalance_ratio(dist)

    weights = compute_class_weights(mat)
    weights.to_csv(output_dir / "class_weights_recommendation.csv", index=False)

    cardinality = df["n_labels"].value_counts().sort_index()
    cardinality.rename_axis("n_labels").reset_index(name="n_records").to_csv(
        output_dir / "label_cardinality.csv", index=False
    )

    plot_class_distribution(dist, output_dir / "class_distribution_overview.png")
    by_fold = plot_distribution_by_fold(df, mat, output_dir / "class_distribution_by_fold.png")
    by_fold.to_csv(output_dir / "class_distribution_by_fold.csv")
    plot_cooccurrence(mat, output_dir / "class_cooccurrence.png")

    write_summary(output_dir, n_total, dist, ratio, max_class, min_class, weights, cardinality)

    print(f"\nListo. Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    main()
