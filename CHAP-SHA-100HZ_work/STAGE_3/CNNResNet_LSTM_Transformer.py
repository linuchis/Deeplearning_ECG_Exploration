"""
=======================================================================
BLOQUE 3 — CHAPMAN-SHAOXING 100 Hz
ETAPA 3 — TRES ARQUITECTURAS DE DEEP LEARNING
=======================================================================
Modelos      : CNN1D_ResNet | CNN_LSTM | Transformer
Tarea        : Clasificación multiclase (AFIB | GSVT | SB | SR)
Entrada      : (batch, 12, 900)   — 12 derivaciones × 900 muestras (9 s
               a 100 Hz, tras el resampleo polifásico 500→100 Hz y el
               recorte de 1 s del transitorio de borde del filtro, ambos
               en STAGE_1_2 — ver cabecera de
               STAGE_1_2/exploration_distribution.py)
Salida       : (batch, 4)         — logits (para CrossEntropyLoss)

Réplica del Bloque 2 (Chapman-Shaoxing 500 Hz) con un único cambio
estructural: la señal es ~5× más corta en número de muestras (900 vs
4500), aunque la MISMA duración en segundos (9 s en ambos, tras el
recorte del transitorio de borde) — así que el bloque de entrada
("CNN stem") de los tres modelos comprime menos agresivamente — ver
"CNN STEM SIMPLIFICADO" abajo.

Diferencias respecto al Bloque 2 (500 Hz):
  · Entrada de 900 muestras en lugar de 4500 (100 Hz vs 500 Hz; misma
    duración de 9 s en ambos bloques)
    → el CNN stem de los tres modelos usa MaxPool ×2 donde el Bloque 2
      usaba ×4, porque la señal ya llega ~5× más corta en muestras (ver
      detalle por modelo en cada clase)
  · batch_size=128 (vuelve al valor del Bloque 1/PTB-XL): con 900
    muestras la señal ocupa un tamaño similar al de PTB-XL (1000), así
    que ya no hace falta reducir el batch a 64 como en el Bloque 2
    (4500 muestras)
  · Pesos de clase cargados desde shaoxing_class_weights.npy generado
    por STAGE_1_2 de este mismo bloque (100 Hz)

Se mantiene igual que en el Bloque 2:
  · Tarea multiclase (1 ritmo por paciente) → CrossEntropyLoss + softmax
  · 4 clases (AFIB, GSVT, SB, SR)
  · División 70/10/20 propia (no strat_fold de PTB-XL)
  · Métrica principal: accuracy + F1 macro (multiclase)

Hipótesis central de la tesis:
  Chapman clasifica RITMOS (dependencias temporales: intervalos RR,
  regularidad del latido) → se espera que CNN-LSTM supere a CNN1D pura,
  al contrario de lo observado en PTB-XL (morfológico). Este Bloque 3
  replica esa hipótesis a 100 Hz para poder comparar directamente contra
  PTB-XL (mismo backbone, misma frecuencia de muestreo) en el Bloque 4.

Justificación por modelo:
  CNN1D_ResNet  → Hannun et al. 2022 (revisión 230 estudios); He et al. 2016
  CNN_LSTM      → Sun et al. 2024 (Sensors, CNN-LSTM-SE, 98.5% acc);
                  Mahajan & Kaul 2025 (GraphEnhanced, CNN-BiLSTM en CS-12)
  Transformer   → Islam et al. 2024 (ECGformer, arXiv 2401.05434, >98%);
                  PatchECG (s41598-026-41549-5, ViT sobre Chapman-Shaoxing)
=======================================================================
"""

import os, sys, csv, json, time
from pathlib import Path

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# =======================================================================
# 0. RUTAS Y CONSTANTES
# =======================================================================

# Arrays preprocesados por STAGE_1_2 de ESTE bloque (100 Hz) — mismo
# criterio que el Bloque 2: se guardan FUERA del proyecto, junto al
# dataset crudo compartido, con sufijo "_100hz" para no chocar con la
# carpeta "preprocessed/" del Bloque 2 (500 Hz).
PREPROCESSED_DIR = Path(
    r"C:\Users\Lina Castañeda\Documents\Originals\CHAPMAN-SHAOXING\preprocessed_100hz"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CLASS_NAMES = ["AFIB", "GSVT", "SB", "SR"]
N_CLASSES   = 4
N_LEADS     = 12
# 9 s a 100 Hz = 900 muestras (resampleadas desde 500 Hz y recortadas 1 s
# por el transitorio de borde del filtro en STAGE_1_2, ver cabecera de
# STAGE_1_2/exploration_distribution.py). FS_TARGET=100 fijado para
# compatibilidad con PTB-XL en el experimento cruzado del Bloque 4.
SIGNAL_LEN  = 900

# ── Paleta fija del proyecto (no generica), consistente con PTBXL_work: ──
# mismo color de arquitectura en TODOS los bloques del proyecto.
PALETTE = ["darkorchid", "seagreen", "royalblue", "darkorange", "firebrick"]
MODEL_COLORS = {
    "CNN1D_ResNet":   PALETTE[0],
    "CNN_LSTM":       PALETTE[1],
    "ECGTransformer": PALETTE[2],
}
# Colormap secuencial para matrices de confusion (blanco -> royalblue),
# igual que en PTBXL_work, en vez de "Blues" generico.
CM_CMAP = LinearSegmentedColormap.from_list("chapman_cm", ["#ffffff", "royalblue"])

TEXTS = {
    "es": {
        "loss_title": "Pérdida (CrossEntropyLoss)",
        "epoch": "Época", "loss_ylabel": "Loss", "train": "train", "val": "val",
        "metrics_title": "Métricas de validación", "score_ylabel": "Score",
        "acc_label": "Accuracy (val)", "f1_label": "F1 macro (val)",
        "curves_suptitle": "{model} — curvas de entrenamiento (Chapman 100 Hz, ventana 9 s)",
        "cm_title": "Matriz de confusión — {model} (test)",
        "predicted": "Predicho", "actual": "Real",
    },
    "en": {
        "loss_title": "Loss (CrossEntropyLoss)",
        "epoch": "Epoch", "loss_ylabel": "Loss", "train": "train", "val": "val",
        "metrics_title": "Validation metrics", "score_ylabel": "Score",
        "acc_label": "Accuracy (val)", "f1_label": "F1 macro (val)",
        "curves_suptitle": "{model} — training curves (Chapman 100 Hz, 9 s window)",
        "cm_title": "Confusion matrix — {model} (test)",
        "predicted": "Predicted", "actual": "Actual",
    },
}


def model_dir(name: str) -> Path:
    return OUTPUT_DIR / name


def model_lang_dir(name: str, lang: str) -> Path:
    return OUTPUT_DIR / name / lang


# =======================================================================
# 1. DATASET Y DATALOADERS
# =======================================================================

class ShaoxingDataset(Dataset):
    """
    Dataset PyTorch para Chapman-Shaoxing (100 Hz).
    Lee los arrays .npy generados en STAGE_1_2 de este bloque.

    __getitem__ devuelve:
      signal : (12, 900) float32   — transpuesta de (900, 12)
               Las Conv1d de PyTorch esperan (batch, canales, longitud)
      label  : int64 escalar       — índice de clase (0-3)
    """
    def __init__(self, split: str):
        X = np.load(PREPROCESSED_DIR / f"shaoxing_{split}_X.npy")
        y = np.load(PREPROCESSED_DIR / f"shaoxing_{split}_y.npy")
        # Transponer: (N, 900, 12) → (N, 12, 900)
        self.X = torch.from_numpy(X.transpose(0, 2, 1))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(batch_size: int = 128,
                      num_workers: int = 4) -> tuple:
    """
    Construye los tres DataLoaders (train/val/test) y carga los pesos
    de clase para CrossEntropyLoss.

    batch_size=128 (vuelve al valor del Bloque 1/PTB-XL):
      A 100 Hz la señal de Chapman (900 muestras) ocupa un tamaño similar
      en VRAM al de PTB-XL (1000), así que ya no aplica la reducción a 64
      que se usó en el Bloque 2 (4500 muestras, 5× más largas).
    """
    train_ds = ShaoxingDataset("train")
    val_ds   = ShaoxingDataset("val")
    test_ds  = ShaoxingDataset("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    # Pesos de clase para CrossEntropyLoss
    class_weights = np.load(PREPROCESSED_DIR / "shaoxing_class_weights.npy")
    class_weights = torch.from_numpy(class_weights).float()

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Pesos de clase: "
          + " ".join(f"{CLASS_NAMES[i]}={class_weights[i]:.4f}"
                     for i in range(N_CLASSES)))
    return train_loader, val_loader, test_loader, class_weights


# =======================================================================
# 2. ARQUITECTURAS
# =======================================================================

# ── 2A. BLOQUE RESIDUAL (idéntico al Bloque 1 y al Bloque 2) ──────────

class ResidualBlock1D(nn.Module):
    """
    Bloque residual 1D — igual que en PTB-XL/Bloque 1 y en el Bloque 2.
    Estructura: Conv → BN → ReLU → Dropout → Conv → BN + skip → ReLU
    """
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1,
                      padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (nn.Sequential(
                         nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                         nn.BatchNorm1d(out_ch))
                     if in_ch != out_ch or stride != 1 else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.skip(x))


# ── 2B. CNN1D RESNET ────────────────────────────────────────────────────

class CNN1D_ResNet(nn.Module):
    """
    CNN 1D ResNet-style adaptada a señales de 900 muestras (100 Hz, 9 s
    tras el recorte del transitorio de borde en STAGE_1_2).

    CNN STEM SIMPLIFICADO (vs. Bloque 2, 500 Hz):
      El Bloque 2 usaba dos MaxPool en la entrada (×4 y ×2) para
      comprimir 4500→562, porque la señal era ~5× más larga en muestras.
      Aquí, con 900 muestras, basta un solo MaxPool ×2 — mismo patrón
      que el bloque de entrada del Bloque 1 (PTB-XL, 1000 muestras).

    Flujo de dimensiones (con batch_size implícito):
      (12, 900)  → input_conv → (64, 450)
                 → res_block1 → (128, 450)
                 → res_block2 → (128, 225)   stride=2
                 → res_block3 → (256, 225)
                 → res_block4 → (256, 113)   stride=2
                 → GAP        → (256, 1)
                 → classifier → (4,)

    Cabeza: CrossEntropyLoss (multiclase, 4 clases)
    """
    def __init__(self, n_leads=12, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 900 → 450
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(64,  128, kernel_size=7, stride=1, dropout=dropout),
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
            # Sin softmax: CrossEntropyLoss lo aplica internamente
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)


# ── 2C. CNN-LSTM ────────────────────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM para clasificación rítmica de ECG.

    Motivación (hipótesis central de la tesis):
      Chapman clasifica RITMOS (AFIB=RR irregular, SB=RR largo,
      GSVT=RR corto y regular, SR=normal). Esas diferencias son
      temporales y secuenciales → la LSTM debería capturarlas mejor
      que una CNN pura.

    CNN STEM SIMPLIFICADO (vs. Bloque 2, 500 Hz):
      El Bloque 2 usaba MaxPool ×4, ×4, ×2 (4500→140) para comprimir la
      señal ~5× más larga en muestras. Aquí, con 900 muestras, cada etapa
      usa MaxPool ×2 (900→112) — mismo patrón de compresión que el
      extractor CNN del Bloque 1 (PTB-XL).

    Arquitectura:
      · Bloque CNN frontal: extrae características locales de morfología
        (ondas P, QRS, T) y reduce la longitud temporal a algo manejable
        para la LSTM (900→112 muestras)
      · LSTM bidireccional: modela la secuencia de características
        a lo largo del tiempo (dependencias de largo alcance)
      · Clasificador FC: proyecta a 4 clases

    Referencia: Sun et al. 2024 (Sensors, CNN-LSTM-SE)
                Mahajan & Kaul 2025 (CNN-BiLSTM en CS-12)
    """
    def __init__(self, n_leads=12, n_classes=N_CLASSES,
                 lstm_hidden=128, lstm_layers=2, dropout=0.3):
        super().__init__()

        # ── Bloque CNN frontal ────────────────────────────────────────
        # Objetivo: extraer características y reducir 900 → 112
        # para que la LSTM no tenga que procesar 900 pasos de tiempo
        self.cnn = nn.Sequential(
            # Capa 1: kernel grande captura morfología completa
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),          # 900 → 450

            # Capa 2
            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),          # 450 → 225

            # Capa 3
            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),          # 225 → 112

            nn.Dropout(dropout),
        )
        # Salida CNN: (batch, 128, 112)

        # ── LSTM bidireccional ────────────────────────────────────────
        # input_size=128 (canales de la CNN)
        # La LSTM recibe la secuencia transpuesta: (batch, 112, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Salida LSTM: (batch, 112, lstm_hidden*2) — bidireccional

        # ── Clasificador ──────────────────────────────────────────────
        # Usamos el último estado oculto (hidden del último paso)
        # que resume toda la secuencia
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        # x: (batch, 12, 900)
        x = self.cnn(x)                    # (batch, 128, 112)
        x = x.permute(0, 2, 1)            # (batch, 112, 128) para LSTM
        _, (h_n, _) = self.lstm(x)        # h_n: (layers*2, batch, hidden)
        # Concatenar último hidden fwd + bwd
        h_fwd = h_n[-2]                   # (batch, hidden)
        h_bwd = h_n[-1]                   # (batch, hidden)
        h = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, hidden*2)
        return self.classifier(h)          # (batch, 4)


# ── 2D. TRANSFORMER ─────────────────────────────────────────────────────

class ECGTransformer(nn.Module):
    """
    Transformer para clasificación de ECG basado en la arquitectura
    patch-based (ECGformer, Islam et al. 2024 / PatchECG).

    Idea clave: en lugar de procesar los 900 pasos uno a uno,
    la señal se divide en PATCHES (ventanas solapadas), cada patch
    se proyecta linealmente como un token, y el Transformer aprende
    relaciones entre patches mediante self-attention.

    Esto es análogo a cómo ViT trata imágenes (patches 2D),
    pero aquí son patches 1D de señal ECG.

    CNN STEM SIMPLIFICADO (vs. Bloque 2, 500 Hz):
      Mismo cambio que en CNN1D_ResNet y CNN_LSTM: cada etapa del stem
      usa MaxPool ×2 en vez de ×4/×4/×2, porque la señal ya llega ~5×
      más corta en muestras (900 vs 4500). La longitud de secuencia
      resultante se sigue calculando dinámicamente (dry run) en vez de
      hardcodearla, para no depender de este valor si SIGNAL_LEN cambia.

    Flujo:
      (batch, 12, signal_len)
        → CNN stem: reduce a (batch, d_model, seq_len) y extrae features locales
        → Patchify: cada posición temporal es un token → (batch, seq_len, d_model)
        → Positional encoding: inyecta posición en la secuencia
        → N capas Transformer Encoder (self-attention + FFN)
        → CLS token o GlobalAvgPool → (batch, d_model)
        → Clasificador FC → (batch, 4)

    Referencia: Islam et al. 2024 (ECGformer, arXiv 2401.05434)
                PatchECG (s41598-026-41549-5)
    """
    def __init__(self, n_leads=12, n_classes=N_CLASSES,
                 d_model=128, n_heads=8, n_layers=4,
                 dim_ff=256, dropout=0.1, signal_len=SIGNAL_LEN):
        super().__init__()

        # ── CNN stem (igual que en CNN_LSTM) ───────────────────────────
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(64, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )
        # Longitud de secuencia de salida del stem: se calcula con un
        # forward "en seco" en vez de hardcodearla (con 900 muestras de
        # entrada y 3 MaxPool ×2, seq_len=112) — evita un mismatch de
        # forma silencioso si SIGNAL_LEN vuelve a cambiar.
        with torch.no_grad():
            seq_len = self.cnn_stem(torch.zeros(1, n_leads, signal_len)).shape[-1]

        # ── CLS token ─────────────────────────────────────────────────
        # Token de clasificación aprendible (como en BERT/ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── Positional encoding (aprendible) ──────────────────────────
        # +1 para el CLS token
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        # ── Transformer Encoder ───────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,    # (batch, seq, features)
            norm_first=True,     # Pre-LN: más estable en entrenamiento
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.dropout = nn.Dropout(dropout)

        # ── Clasificador ──────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x: (batch, 12, signal_len)
        B = x.size(0)

        x = self.cnn_stem(x)              # (B, d_model, seq_len)
        x = x.permute(0, 2, 1)           # (B, seq_len, d_model)

        # Añadir CLS token al inicio de la secuencia
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, seq_len+1, d_model)

        # Positional encoding
        x = self.dropout(x + self.pos_embedding)

        # Transformer encoder
        x = self.transformer(x)           # (B, seq_len+1, d_model)

        # Usar el CLS token para clasificar
        cls_out = x[:, 0]                 # (B, 128)

        return self.classifier(cls_out)   # (B, 4)


# =======================================================================
# 3. MÉTRICAS (multiclase — diferente a Bloque 1 que era multi-label)
# =======================================================================

def compute_metrics(all_labels: np.ndarray,
                    all_preds:  np.ndarray,
                    all_probs:  np.ndarray) -> dict:
    """
    Métricas para clasificación multiclase (4 clases mutuamente excluyentes).

    Diferencia respecto al Bloque 1:
      PTB-XL  → AUC-ROC macro (umbral variable, multi-label)
      Chapman → Accuracy + F1 macro (argmax de softmax, multiclase)

    all_labels : (N,)    int  — clase verdadera
    all_preds  : (N,)    int  — clase predicha (argmax)
    all_probs  : (N, 4) float — probabilidades softmax
    """
    metrics = {}
    metrics["accuracy"]  = float(accuracy_score(all_labels, all_preds))
    metrics["f1_macro"]  = float(f1_score(all_labels, all_preds,
                                           average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(all_labels, all_preds,
                                             average="weighted", zero_division=0))

    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for cls, f1 in zip(CLASS_NAMES, f1_per):
        metrics[f"f1_{cls}"] = float(f1)

    metrics["classification_report"] = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(
        all_labels, all_preds
    ).tolist()

    return metrics


# =======================================================================
# 4. BUCLE DE ENTRENAMIENTO
# =======================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits = model(signals)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    all_probs  = np.vstack(all_probs)

    metrics  = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / len(loader)
    return avg_loss, metrics


# =======================================================================
# 5. ENTRENAMIENTO COMPLETO (reutilizable para los 3 modelos)
# =======================================================================

def train_model(model_name:    str,
                model:         nn.Module,
                train_loader:  DataLoader,
                val_loader:    DataLoader,
                class_weights: torch.Tensor,
                n_epochs:      int   = 50,
                lr:            float = 1e-3) -> tuple:
    """
    Entrena cualquiera de los tres modelos con la misma configuración:
      · CrossEntropyLoss con pesos de clase
      · Adam + weight_decay=1e-4
      · ReduceLROnPlateau (métrica: F1 macro val)
      · Early stopping (patience=10, métrica: F1 macro val)
      · Guarda el mejor checkpoint según F1 macro val

    Métrica de monitoreo: F1 macro (multiclase)
    En PTB-XL era AUC macro (multi-label). El cambio es coherente
    con el tipo de tarea.
    """
    model_dir(model_name).mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir(model_name) / f"best_{model_name}.pt"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")
    print(f"  Dispositivo  : {device}")
    print(f"  Parámetros   : {n_params:,}")
    print(f"  Épocas máx.  : {n_epochs}")
    print(f"  LR inicial   : {lr}")
    print(f"  Loss         : CrossEntropyLoss (multiclase, pesos de clase)")
    print(f"  Early stop   : patience=10, métrica=F1 macro val")
    print("-" * 60)

    best_f1, patience_cnt, patience_max = 0.0, 0, 10
    history = {"train_loss": [], "val_loss": [],
               "val_acc": [],    "val_f1": []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss            = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device)

        val_acc = val_metrics["accuracy"]
        val_f1  = val_metrics["f1_macro"]

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            print(f"  ↓ LR reducido: {lr_before:.2e} → {lr_after:.2e}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Época {epoch:02d}/{n_epochs} | "
              f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"Acc={val_acc:.4f} F1={val_f1:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     val_acc,
                "val_f1":      val_f1,
                "history":     history,
            }, checkpoint_path)
            print(f"  ✓ Mejor modelo guardado (F1={best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience_max:
                print(f"\nEarly stopping en época {epoch}.")
                break

    print(f"\nMejor F1 macro en validación: {best_f1:.4f}")
    return model, history, str(checkpoint_path)


# =======================================================================
# 6. EVALUACIÓN FINAL EN TEST
# =======================================================================

def evaluate_on_test(model_name:    str,
                     model_class,
                     model_kwargs:  dict,
                     test_loader:   DataLoader,
                     class_weights: torch.Tensor) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model_class(**model_kwargs).to(device)
    ckpt   = torch.load(model_dir(model_name) / f"best_{model_name}.pt",
                        map_location=device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"  RESULTADOS TEST — {model_name}")
    print(f"{'='*60}")
    print(f"  Loss      : {test_loss:.4f}")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  F1 macro  : {test_metrics['f1_macro']:.4f}")
    print(f"  F1 weighted: {test_metrics['f1_weighted']:.4f}")
    print("\n  F1 por clase:")
    for cls in CLASS_NAMES:
        print(f"    {cls:<6}: {test_metrics[f'f1_{cls}']:.4f}")
    print(f"\n{test_metrics['classification_report']}")

    test_metrics["loss"] = float(test_loss)
    return test_metrics


# =======================================================================
# 7. VISUALIZACIONES
# =======================================================================

def save_history_csv(history: dict, path: Path):
    keys = list(history.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history[keys[0]])):
            writer.writerow([i + 1] + [history[k][i] for k in keys])


def plot_training_curves(model_name: str, history: dict, lang: str, out_dir: Path):
    t_ = TEXTS[lang]
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label=t_["train"], color=PALETTE[0])
    axes[0].plot(epochs, history["val_loss"],   label=t_["val"],   color=PALETTE[1])
    axes[0].set_title(t_["loss_title"])
    axes[0].set_xlabel(t_["epoch"]); axes[0].set_ylabel(t_["loss_ylabel"])
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label=t_["acc_label"], color=PALETTE[3])
    axes[1].plot(epochs, history["val_f1"],  label=t_["f1_label"],  color=PALETTE[4])
    axes[1].set_title(t_["metrics_title"])
    axes[1].set_xlabel(t_["epoch"]); axes[1].set_ylabel(t_["score_ylabel"])
    axes[1].set_ylim(0, 1); axes[1].legend()

    fig.suptitle(t_["curves_suptitle"].format(model=model_name))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model_name: str, test_metrics: dict, lang: str, out_dir: Path):
    t_ = TEXTS[lang]
    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=CM_CMAP,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel(t_["predicted"]); ax.set_ylabel(t_["actual"])
    ax.set_title(t_["cm_title"].format(model=model_name))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_test_report(model_name: str, test_metrics: dict,
                     json_path: Path, txt_path_es: Path, txt_path_en: Path):
    numeric = {k: v for k, v in test_metrics.items()
               if k not in ("classification_report", "confusion_matrix")}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(numeric, f, indent=2)

    lines_es = [
        "=" * 60,
        f"RESULTADOS TEST — {model_name} (Chapman-Shaoxing 100 Hz, ventana 9 s)",
        "=" * 60,
        f"Loss      : {test_metrics['loss']:.4f}",
        f"Accuracy  : {test_metrics['accuracy']:.4f}",
        f"F1 macro  : {test_metrics['f1_macro']:.4f}",
        f"F1 wt     : {test_metrics['f1_weighted']:.4f}",
        "",
        "F1 por clase:",
    ] + [f"  {c:<6}: {test_metrics[f'f1_{c}']:.4f}" for c in CLASS_NAMES] + [
        "",
        "Reporte de clasificación (umbral=argmax):",
        test_metrics["classification_report"],
        "=" * 60,
    ]
    txt_path_es.write_text("\n".join(lines_es), encoding="utf-8")

    lines_en = [
        "=" * 60,
        f"TEST RESULTS — {model_name} (Chapman-Shaoxing 100 Hz, 9 s window)",
        "=" * 60,
        f"Loss      : {test_metrics['loss']:.4f}",
        f"Accuracy  : {test_metrics['accuracy']:.4f}",
        f"F1 macro  : {test_metrics['f1_macro']:.4f}",
        f"F1 wt     : {test_metrics['f1_weighted']:.4f}",
        "",
        "F1 per class:",
    ] + [f"  {c:<6}: {test_metrics[f'f1_{c}']:.4f}" for c in CLASS_NAMES] + [
        "",
        "Classification report (threshold=argmax):",
        test_metrics["classification_report"],
        "=" * 60,
    ]
    txt_path_en.write_text("\n".join(lines_en), encoding="utf-8")


# =======================================================================
# 8. MAIN
# =======================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ──────────────────────────────────────────────────
    # batch_size=128: señal de tamaño similar a PTB-XL (900 vs 1000 muestras)
    # num_workers=4: misma configuración que Bloque 1 y Bloque 2 (Ryzen 5 5600X)
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        batch_size=128, num_workers=4
    )

    # ── Definición de los tres modelos ───────────────────────────────
    MODELOS = [
        {
            "name":   "CNN1D_ResNet",
            "class":  CNN1D_ResNet,
            "kwargs": {"n_leads": 12, "n_classes": N_CLASSES, "dropout": 0.3},
        },
        {
            "name":   "CNN_LSTM",
            "class":  CNN_LSTM,
            "kwargs": {"n_leads": 12, "n_classes": N_CLASSES,
                       "lstm_hidden": 128, "lstm_layers": 2, "dropout": 0.3},
        },
        {
            "name":   "ECGTransformer",
            "class":  ECGTransformer,
            "kwargs": {"n_leads": 12, "n_classes": N_CLASSES,
                       "d_model": 128, "n_heads": 8, "n_layers": 4,
                       "dim_ff": 256, "dropout": 0.1},
        },
    ]

    all_test_results = {}

    for cfg in MODELOS:
        name = cfg["name"]
        model = cfg["class"](**cfg["kwargs"])

        mdir    = model_dir(name)
        es_dir  = model_lang_dir(name, "es")
        en_dir  = model_lang_dir(name, "en")
        mdir.mkdir(parents=True, exist_ok=True)
        es_dir.mkdir(parents=True, exist_ok=True)
        en_dir.mkdir(parents=True, exist_ok=True)

        # ── Entrenamiento ─────────────────────────────────────────────
        _, history, ckpt_path = train_model(
            model_name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            n_epochs=50,
            lr=1e-3,
        )

        # ── Guardar historial y curvas (curvas: es/en) ─────────────────
        save_history_csv(history, mdir / "history.csv")
        plot_training_curves(name, history, lang="es", out_dir=es_dir)
        plot_training_curves(name, history, lang="en", out_dir=en_dir)

        # ── Evaluación en test ────────────────────────────────────────
        test_metrics = evaluate_on_test(
            model_name=name,
            model_class=cfg["class"],
            model_kwargs=cfg["kwargs"],
            test_loader=test_loader,
            class_weights=class_weights,
        )

        # ── Guardar resultados (json neutral, txt/confusion: es/en) ────
        save_test_report(
            name, test_metrics,
            json_path=mdir / "test_metrics.json",
            txt_path_es=es_dir / "test_metrics.txt",
            txt_path_en=en_dir / "test_metrics.txt",
        )
        plot_confusion_matrix(name, test_metrics, lang="es", out_dir=es_dir)
        plot_confusion_matrix(name, test_metrics, lang="en", out_dir=en_dir)
        all_test_results[name] = test_metrics

    # ── Tabla comparativa final ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARATIVA FINAL — CHAPMAN-SHAOXING 100 Hz")
    print(f"{'='*60}")
    print(f"  {'Modelo':<18} {'Accuracy':>9} {'F1 macro':>9} {'F1 AFIB':>8} "
          f"{'F1 GSVT':>8} {'F1 SB':>7} {'F1 SR':>7}")
    print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for name, m in all_test_results.items():
        print(f"  {name:<18} {m['accuracy']:>9.4f} {m['f1_macro']:>9.4f} "
              f"{m['f1_AFIB']:>8.4f} {m['f1_GSVT']:>8.4f} "
              f"{m['f1_SB']:>7.4f} {m['f1_SR']:>7.4f}")
    print(f"{'='*60}")

    with open(OUTPUT_DIR / "comparativa_final.json", "w",
              encoding="utf-8") as f:
        json.dump({n: {k: v for k, v in m.items()
                       if k not in ("classification_report",
                                    "confusion_matrix")}
                   for n, m in all_test_results.items()},
                  f, indent=2)

    print(f"\n✓ Etapa 3 del Bloque 3 completada.")
    print(f"  Outputs en: {OUTPUT_DIR}")
