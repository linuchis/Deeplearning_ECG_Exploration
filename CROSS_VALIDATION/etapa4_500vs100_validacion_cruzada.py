# =============================================================================
# BLOQUE 3 — ETAPA 4
# Parte A: Comparativa Chapman-Shaoxing 500 Hz vs 100 Hz
# Parte B: Validación cruzada PTB-XL → Chapman-Shaoxing 100 Hz
# =============================================================================
# Script autocontenido: no importa nada de otros scripts del proyecto (las
# arquitecturas de PTB-XL se REDEFINEN aquí, copiadas de PTBXL_work/STAGE_3,
# para no depender de sys.path ni de imports cruzados entre bloques). Los
# únicos archivos externos que lee son los .npy de Chapman 100 Hz y los
# checkpoints .pt de PTB-XL.
#
# PARTE A — Comparativa 500 Hz vs 100 Hz:
#   Métricas hardcodeadas (test set), tomadas de:
#     CHAP-SHA-500HZ_work/STAGE_3/outputs/comparativa_final.json
#     CHAP-SHA-100HZ_work/STAGE_3/outputs/comparativa_final.json
#   4 figuras: barras agrupadas F1 macro, diferencia 500−100 por clase,
#   heatmap de las 6 métricas, y slope chart del ranking entre frecuencias.
#
# PARTE B — Validación cruzada PTB-XL ↔ Chapman:
#   Carga los mejores checkpoints de los 3 modelos entrenados en PTB-XL
#   (5 superclases: NORM|MI|STTC|CD|HYP, multi-label, BCEWithLogitsLoss) y
#   los evalúa DIRECTAMENTE sobre el test set de Chapman 100 Hz (4 ritmos:
#   AFIB|GSVT|SB|SR), sin reentrenar ni fine-tunear.
#
#   Los espacios de clases son incompatibles (PTB-XL clasifica morfología
#   de infarto/hipertrofia; Chapman clasifica ritmo cardíaco) — por eso
#   NO se calcula accuracy/F1 "cruzados" contra ninguna etiqueta de Chapman
#   forzada a las 5 clases de PTB-XL, ya que esa comparación no tendría
#   ningún referente clínico válido. En su lugar, esto es un análisis
#   EXPLORATORIO de cómo "reacciona" un clasificador entrenado en PTB-XL
#   al ver una distribución de señales completamente distinta:
#     · Distribución de la clase más probable (argmax del sigmoid) por
#       modelo, sobre las 2118 señales de test de Chapman.
#     · Probabilidad media (sigmoid) por clase de PTB-XL, por modelo.
#     · Entropía de Shannon de la distribución de probabilidades por
#       señal (normalizada a sumar 1, ya que sigmoid multi-label no suma
#       1 por diseño) — mide qué tan "seguro" o "disperso" está cada
#       modelo al enfrentar señales fuera de su distribución de
#       entrenamiento. No se usan etiquetas ficticias de Chapman para
#       esto: la entropía se calcula sobre la propia salida del modelo,
#       sin ninguna referencia externa (por eso no aplica una pérdida de
#       entropía cruzada con "etiquetas ficticias" — no hay etiqueta
#       ficticia que tenga sentido clínico aquí, así que se reporta
#       únicamente la autoentropía de la predicción).
#
#   Mismatch de longitud (PTB-XL: 1000 muestras @ 100 Hz | Chapman 100 Hz:
#   900 muestras @ 100 Hz, tras el recorte del transitorio de borde en
#   STAGE_1_2): se resuelve con zero-padding al final de la señal,
#   np.pad(X, ((0,0),(0,100),(0,0))) → (N, 1000, 12), exactamente como se
#   especificó. Esto NO reintroduce el segundo recortado — son 100
#   muestras de ceros al final, no la señal descartada.
# =============================================================================

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# 0. RUTAS Y CONSTANTES
# =============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "etapa4_500vs100"

# Checkpoints de PTB-XL (Bloque 1) — ver PTBXL_work/STAGE_3
PTBXL_STAGE3_DIR = Path(r"C:\Users\Lina Castañeda\Deeplearning_ECG_Exploration\PTBXL_work\STAGE_3")
CKPT_CNN1D        = PTBXL_STAGE3_DIR / "CNN_1D"        / "outputs" / "best_cnn1d.pt"
CKPT_CNN_LSTM     = PTBXL_STAGE3_DIR / "CNN-LSTM-V2"    / "outputs" / "best_cnn_lstm_v2.pt"
CKPT_TRANSFORMER  = PTBXL_STAGE3_DIR / "TRANSFORMER-V2" / "outputs" / "best_transformer_v2.pt"

# Test set de Chapman 100 Hz (Bloque 3) — ver ubicación fijada en
# CHAP-SHA-100HZ_work/STAGE_1_2/exploration_distribution.py
# (PREPROCESSED_OUTPUT_DIR): vive fuera del repo, junto al dataset crudo.
CHAPMAN_100HZ_PREPROCESSED = Path(
    r"C:\Users\Lina Castañeda\Documents\Originals\CHAPMAN-SHAOXING\preprocessed_100hz"
)
CHAPMAN_TEST_X = CHAPMAN_100HZ_PREPROCESSED / "shaoxing_test_X.npy"

# Colores fijos por modelo (dados por especificación, consistentes en
# ambas partes del script)
MODEL_COLORS = {
    "CNN1D_ResNet":   "#E63946",
    "CNN_LSTM":       "#F4A261",
    "ECGTransformer": "#2A9D8F",
}
MODEL_ORDER = ["CNN1D_ResNet", "CNN_LSTM", "ECGTransformer"]

PTBXL_CLASSES   = ["NORM", "MI", "STTC", "CD", "HYP"]
CHAPMAN_CLASSES = ["AFIB", "GSVT", "SB", "SR"]


# =============================================================================
# PARTE A — COMPARATIVA CHAPMAN 500 Hz vs 100 Hz
# =============================================================================

# ── A.0  Resultados leídos en vivo desde los comparativa_final.json ───────
# (antes iban hardcodeados como snapshot; se desalinearon del resultado real
# en cuanto se reentrenó 500 Hz tras el fix del Z-score/NaN de Etapa 2 —
# por eso ahora se leen directo de los JSON en cada corrida)

COMPARATIVA_500HZ = Path(
    r"C:\Users\Lina Castañeda\Deeplearning_ECG_Exploration\CHAP-SHA-500HZ_work\STAGE_3\outputs\comparativa_final.json"
)
COMPARATIVA_100HZ = Path(
    r"C:\Users\Lina Castañeda\Deeplearning_ECG_Exploration\CHAP-SHA-100HZ_work\STAGE_3\outputs\comparativa_final.json"
)


def cargar_resultados_500_100() -> dict:
    with open(COMPARATIVA_500HZ, "r", encoding="utf-8") as f:
        r500 = json.load(f)
    with open(COMPARATIVA_100HZ, "r", encoding="utf-8") as f:
        r100 = json.load(f)
    return {"500 Hz": r500, "100 Hz": r100}


RESULTS = cargar_resultados_500_100()

METRICS = ["accuracy", "f1_macro", "f1_AFIB", "f1_GSVT", "f1_SB", "f1_SR"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "f1_macro": "F1 macro",
    "f1_AFIB": "F1 AFIB", "f1_GSVT": "F1 GSVT",
    "f1_SB": "F1 SB", "f1_SR": "F1 SR",
}


def fig1_barras_f1_macro(out_dir: Path):
    """Fig 1: barras agrupadas de F1 macro — 3 modelos × 2 frecuencias."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.32

    freqs = [("500 Hz", None, 1.0), ("100 Hz", "//", 0.55)]
    for i, (freq, hatch, alpha) in enumerate(freqs):
        offset = (i - 0.5) * width
        vals = [RESULTS[freq][m]["f1_macro"] for m in MODEL_ORDER]
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
        bars = ax.bar(x + offset, vals, width, color=colors, alpha=alpha,
                      hatch=hatch, edgecolor="#2b2b2b", linewidth=0.8,
                      label=freq)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("F1 macro (test)")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Chapman-Shaoxing — F1 macro por modelo: 500 Hz vs 100 Hz")
    ax.legend(title="Frecuencia")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = out_dir / "fig1_f1_macro_barras.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def fig2_diferencia_500_100(out_dir: Path):
    """Fig 2: diferencia (500 Hz − 100 Hz) por modelo y por clase, con línea en 0."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(CHAPMAN_CLASSES))
    width = 0.25

    for i, model in enumerate(MODEL_ORDER):
        offset = (i - 1) * width
        diffs = [
            RESULTS["500 Hz"][model][f"f1_{cls}"] - RESULTS["100 Hz"][model][f"f1_{cls}"]
            for cls in CHAPMAN_CLASSES
        ]
        bars = ax.bar(x + offset, diffs, width, color=MODEL_COLORS[model],
                      edgecolor="#2b2b2b", linewidth=0.6, label=model)
        ax.bar_label(bars, fmt="%+.3f", padding=2, fontsize=7, rotation=90)

    ax.axhline(0, color="#2b2b2b", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(CHAPMAN_CLASSES)
    ax.set_ylabel("Δ F1 (500 Hz − 100 Hz)")
    ax.set_title("Chapman-Shaoxing — diferencia de F1 por clase (500 Hz − 100 Hz)\n"
                 "positivo = mejor a 500 Hz · negativo = mejor a 100 Hz")
    ax.legend(title="Modelo")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = out_dir / "fig2_diferencia_500_100.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def fig3_heatmap_metricas(out_dir: Path):
    """Fig 3: heatmap con las 6 métricas × (3 modelos × 2 frecuencias)."""
    columns = [(m, f) for m in MODEL_ORDER for f in ("500 Hz", "100 Hz")]
    data = np.array([
        [RESULTS[freq][model][metric] for (model, freq) in columns]
        for metric in METRICS
    ])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    vmin, vmax = data.min(), data.max()
    im = ax.imshow(data, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            color = "white" if (val - vmin) / (vmax - vmin) > 0.6 else "#2b2b2b"
            ax.text(c, r, f"{val:.4f}", ha="center", va="center",
                    fontsize=8.5, color=color)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{m}\n{f}" for m, f in columns], fontsize=8)
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_title("Chapman-Shaoxing — las 6 métricas × 3 modelos × 2 frecuencias (test)")

    # Separadores verticales entre modelos (cada 2 columnas)
    for i in range(1, len(MODEL_ORDER)):
        ax.axvline(i * 2 - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score")

    fig.tight_layout()
    out = out_dir / "fig3_heatmap_metricas.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def fig4_slope_ranking(out_dir: Path):
    """Fig 4: slope chart del ranking de modelos (F1 macro) entre 500 Hz y 100 Hz."""
    fig, ax = plt.subplots(figsize=(7, 6))
    x = [0, 1]

    for model in MODEL_ORDER:
        y500 = RESULTS["500 Hz"][model]["f1_macro"]
        y100 = RESULTS["100 Hz"][model]["f1_macro"]
        ax.plot(x, [y500, y100], marker="o", markersize=8,
                color=MODEL_COLORS[model], linewidth=2.5, label=model)
        ax.annotate(f"{model}\n{y500:.4f}", (x[0], y500),
                    xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=MODEL_COLORS[model])
        ax.annotate(f"{model}\n{y100:.4f}", (x[1], y100),
                    xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, color=MODEL_COLORS[model])

    # Ranking (1=mejor) anotado a un lado de cada punto
    for xi, freq in zip(x, ("500 Hz", "100 Hz")):
        ranked = sorted(MODEL_ORDER, key=lambda m: -RESULTS[freq][m]["f1_macro"])
        for rank, model in enumerate(ranked, start=1):
            y = RESULTS[freq][model]["f1_macro"]
            ax.text(xi, y, f"  #{rank}" if xi == 1 else f"#{rank}  ",
                    ha="left" if xi == 1 else "right", va="center",
                    fontsize=8, fontweight="bold", color="#2b2b2b")

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["500 Hz", "100 Hz"], fontsize=11)
    ax.set_ylabel("F1 macro (test)")
    ax.set_title("Chapman-Shaoxing — ranking de modelos por F1 macro\n500 Hz → 100 Hz")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    out = out_dir / "fig4_slope_ranking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def parte_a_comparativa(out_dir: Path):
    print("\n" + "=" * 60)
    print("  PARTE A — Comparativa Chapman-Shaoxing 500 Hz vs 100 Hz")
    print("=" * 60)

    fig1_barras_f1_macro(out_dir)
    fig2_diferencia_500_100(out_dir)
    fig3_heatmap_metricas(out_dir)
    fig4_slope_ranking(out_dir)

    with open(out_dir / "comparativa_500vs100.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print(f"[OK] {out_dir / 'comparativa_500vs100.json'}")


# =============================================================================
# PARTE B — VALIDACIÓN CRUZADA PTB-XL → CHAPMAN 100 Hz
# =============================================================================
# Arquitecturas copiadas de PTBXL_work/STAGE_3 (CNN_1D/model1_cnn1d.py,
# CNN-LSTM-V2/model2_cnn_v2.py, TRANSFORMER-V2/transformer-v2.py) para que
# este script sea autocontenido. Los hiperparámetros de construcción
# coinciden EXACTAMENTE con los usados al entrenar cada checkpoint.

# ── B.1  CNN1D_ResNet ───────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))


class CNN1D_ResNet(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, dropout=0.3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(64, 128, kernel_size=7, stride=1, dropout=dropout),
            ResidualBlock1D(128, 128, kernel_size=7, stride=2, dropout=dropout),
            ResidualBlock1D(128, 256, kernel_size=5, stride=1, dropout=dropout),
            ResidualBlock1D(256, 256, kernel_size=5, stride=2, dropout=dropout),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


# ── B.2  CNN_LSTM (V2, readout max-pool) ────────────────────────────────

class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_leads=12, dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x.permute(0, 2, 1)


class CNN_LSTM(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = ConvFeatureExtractor(n_leads=n_leads, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=256, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_size = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        features = self.cnn(x)
        lstm_out, _ = self.lstm(features)
        pooled = lstm_out.max(dim=1).values   # readout V2: max-pool temporal
        return self.classifier(pooled)


# ── B.3  ECGTransformer (V2, regularizado) ──────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, n_leads=12, patch_size=50, d_model=256, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.local_cnn = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.patch_proj = nn.Conv1d(64, d_model, kernel_size=patch_size, stride=patch_size, bias=False)
        self.patch_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.local_cnn(x)
        x = self.patch_proj(x)
        x = x.permute(0, 2, 1)
        x = self.patch_norm(x)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_patches, d_model, dropout=0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ECGTransformer(nn.Module):
    def __init__(self, n_leads=12, n_classes=5, patch_size=50, d_model=256,
                 n_heads=8, n_layers=2, mlp_ratio=4, dropout=0.3):
        super().__init__()
        n_patches = 1000 // patch_size
        self.patch_embed = PatchEmbedding(n_leads=n_leads, patch_size=patch_size,
                                          d_model=d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_enc = PositionalEncoding(max_patches=n_patches + 1, d_model=d_model, dropout=dropout)
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.norm(x[:, 0, :])
        return self.classifier(cls_out)


# ── B.4  Carga de checkpoints + inferencia ──────────────────────────────

def build_ptbxl_models() -> tuple:
    """Instancia los 3 modelos con los MISMOS hiperparámetros usados al
    entrenarlos en PTB-XL, y les carga su mejor checkpoint (state_dict
    bajo la clave "model_state", igual convención en los 3 scripts de
    PTBXL_work/STAGE_3)."""
    specs = {
        "CNN1D_ResNet":   (CNN1D_ResNet(n_leads=12, n_classes=5, dropout=0.3), CKPT_CNN1D),
        "CNN_LSTM":       (CNN_LSTM(n_leads=12, n_classes=5, hidden_size=256,
                                    num_layers=2, dropout=0.3), CKPT_CNN_LSTM),
        "ECGTransformer": (ECGTransformer(n_leads=12, n_classes=5, patch_size=50,
                                          d_model=256, n_heads=8, n_layers=2,
                                          mlp_ratio=4, dropout=0.3), CKPT_TRANSFORMER),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for name, (model, ckpt_path) in specs.items():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        models[name] = model.to(device).eval()
        print(f"[OK] {name}: checkpoint cargado de {ckpt_path.name}")
    return models, device


def cargar_chapman_test_padded() -> torch.Tensor:
    """Carga el test set de Chapman 100 Hz (N, 900, 12), lo rellena con
    ceros al final hasta 1000 muestras (longitud esperada por PTB-XL) y
    lo transpone a (N, 12, 1000), igual formato que en entrenamiento."""
    X = np.load(CHAPMAN_TEST_X)
    print(f"[INFO] Chapman test X cargado: {X.shape}")
    assert X.shape[1:] == (900, 12), f"Shape inesperado: {X.shape}"

    X_padded = np.pad(X, ((0, 0), (0, 100), (0, 0)))   # (N, 900, 12) → (N, 1000, 12)
    assert X_padded.shape[1:] == (1000, 12)

    X_t = torch.from_numpy(X_padded.transpose(0, 2, 1)).float()   # (N, 12, 1000)
    print(f"[OK] Zero-padding aplicado: {X.shape} → {tuple(X_t.shape)} (batch, leads, tiempo)")
    return X_t


@torch.no_grad()
def run_inference(model: nn.Module, X_t: torch.Tensor, device, batch_size: int = 256) -> np.ndarray:
    """Corre el modelo por lotes y devuelve las probabilidades sigmoid
    (N, 5) — sin backward, sin optimizador: es solo evaluación."""
    all_probs = []
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i + batch_size].to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def analizar_predicciones(probs: np.ndarray) -> dict:
    """Análisis exploratorio de las predicciones de un modelo de PTB-XL
    sobre señales de Chapman (ver cabecera del script para la
    justificación de por qué no se reportan métricas de clasificación).

    · argmax_distribution : a qué clase de PTB-XL "se parece más" cada
      señal de Chapman, según el modelo (clase con mayor sigmoid).
    · mean_sigmoid_per_class : probabilidad media por clase (0-1),
      independiente para cada una de las 5 salidas (esquema multi-label).
    · entropy_mean/std : entropía de Shannon de la distribución de
      probabilidades por señal, normalizada a sumar 1 (las 5 salidas
      sigmoid no suman 1 por construcción, al ser multi-label — se
      renormalizan solo para poder calcular una entropía interpretable).
      Entropía alta → el modelo está "indeciso" entre varias clases de
      PTB-XL al ver la señal de Chapman (esperable, al ser datos fuera
      de su distribución de entrenamiento). No se usa ninguna etiqueta
      real ni ficticia de Chapman: es autoentropía de la predicción.
    """
    argmax_idx = probs.argmax(axis=1)
    counts = np.bincount(argmax_idx, minlength=len(PTBXL_CLASSES))
    argmax_distribution = {cls: int(c) for cls, c in zip(PTBXL_CLASSES, counts)}
    argmax_distribution_pct = {cls: float(c) / len(probs) for cls, c in argmax_distribution.items()}

    mean_sigmoid_per_class = {cls: float(probs[:, i].mean())
                              for i, cls in enumerate(PTBXL_CLASSES)}

    p_norm = probs / probs.sum(axis=1, keepdims=True)
    entropy = -(p_norm * np.log(p_norm + 1e-12)).sum(axis=1)

    return {
        "n_signals": int(len(probs)),
        "argmax_distribution": argmax_distribution,
        "argmax_distribution_pct": argmax_distribution_pct,
        "mean_sigmoid_per_class": mean_sigmoid_per_class,
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
    }


def fig5_validacion_cruzada(resultados: dict, out_dir: Path):
    """Distribución de la clase argmax (izquierda) y entropía media de
    predicción (derecha) por modelo, sobre el test set de Chapman."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel izquierdo: distribución argmax por clase de PTB-XL, por modelo
    x = np.arange(len(PTBXL_CLASSES))
    width = 0.25
    for i, model in enumerate(MODEL_ORDER):
        offset = (i - 1) * width
        vals = [resultados[model]["argmax_distribution_pct"][cls] for cls in PTBXL_CLASSES]
        bars = axes[0].bar(x + offset, vals, width, color=MODEL_COLORS[model],
                           edgecolor="#2b2b2b", linewidth=0.6, label=model)
        axes[0].bar_label(bars, fmt="%.2f", padding=2, fontsize=7, rotation=90)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(PTBXL_CLASSES)
    axes[0].set_ylabel("Proporción de señales de Chapman")
    axes[0].set_title("Clase de PTB-XL más probable (argmax)\npor señal de Chapman — test 100 Hz")
    axes[0].legend(title="Modelo (entrenado en PTB-XL)", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel derecho: entropía media ± std por modelo
    means = [resultados[m]["entropy_mean"] for m in MODEL_ORDER]
    stds = [resultados[m]["entropy_std"] for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = axes[1].bar(MODEL_ORDER, means, yerr=stds, capsize=5,
                       color=colors, edgecolor="#2b2b2b", linewidth=0.6)
    axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    axes[1].axhline(np.log(len(PTBXL_CLASSES)), color="#2b2b2b", linewidth=1.0,
                    linestyle="--", label=f"máx. teórica (ln {len(PTBXL_CLASSES)})")
    axes[1].set_ylabel("Entropía de Shannon (nats)")
    axes[1].set_title("Entropía media de la predicción\n(autoentropía, sin etiquetas de Chapman)")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Validación cruzada PTB-XL → Chapman-Shaoxing 100 Hz (análisis exploratorio)")
    fig.tight_layout()
    out = out_dir / "fig5_validacion_cruzada.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def parte_b_validacion_cruzada(out_dir: Path):
    print("\n" + "=" * 60)
    print("  PARTE B — Validación cruzada PTB-XL → Chapman 100 Hz")
    print("=" * 60)

    models, device = build_ptbxl_models()
    print(f"[INFO] Dispositivo: {device}")

    X_t = cargar_chapman_test_padded()

    resultados = {}
    for name, model in models.items():
        probs = run_inference(model, X_t, device)
        resultados[name] = analizar_predicciones(probs)
        r = resultados[name]
        print(f"\n  {name}:")
        print(f"    Distribución argmax : {r['argmax_distribution']}")
        print(f"    Entropía media±std  : {r['entropy_mean']:.4f} ± {r['entropy_std']:.4f}")

    fig5_validacion_cruzada(resultados, out_dir)

    with open(out_dir / "validacion_cruzada_resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"[OK] {out_dir / 'validacion_cruzada_resultados.json'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parte_a_comparativa(OUTPUT_DIR)
    parte_b_validacion_cruzada(OUTPUT_DIR)

    print("\n✓ Etapa 4 completada.")
    print(f"  Outputs en: {OUTPUT_DIR}")
