"""
=======================================================================
BLOQUE 2 — CHAPMAN-SHAOXING
ETAPA 3 — TRES ARQUITECTURAS DE DEEP LEARNING
=======================================================================
Modelos      : CNN1D_ResNet | CNN_LSTM | Transformer
Tarea        : Clasificación multiclase (AFIB | GSVT | SB | SR)
Entrada      : (batch, 12, 5000)  — 12 derivaciones × 5000 muestras
Salida       : (batch, 4)         — logits (para CrossEntropyLoss)

Diferencias respecto al Bloque 1 (PTB-XL):
  · Tarea multiclase (1 ritmo por paciente) → CrossEntropyLoss + softmax
    En PTB-XL era multi-label → BCEWithLogitsLoss + sigmoid
  · 4 clases en lugar de 5 superclases
  · Entrada de 5000 muestras en lugar de 1000 (500 Hz vs 100 Hz)
    → Los bloques de pooling comprimen más agresivamente la secuencia
  · Pesos de clase cargados desde shaoxing_class_weights.npy
    (mismo concepto que pos_weights en Bloque 1, distinta forma de cálculo)
  · División 70/10/20 propia (no strat_fold de PTB-XL)
  · Métrica principal: accuracy + F1 macro (multiclase)
    En PTB-XL era AUC-ROC macro (multi-label)

Hipótesis central de la tesis:
  Chapman clasifica RITMOS (dependencias temporales: intervalos RR,
  regularidad del latido) → se espera que CNN-LSTM supere a CNN1D pura,
  al contrario de lo observado en PTB-XL (morfológico).

Justificación por modelo:
  CNN1D_ResNet  → Hannun et al. 2022 (revisión 230 estudios); He et al. 2016
  CNN_LSTM      → Sun et al. 2024 (Sensors, CNN-LSTM-SE, 98.5% acc);
                  Mahajan & Kaul 2025 (GraphEnhanced, CNN-BiLSTM en CS-12)
  Transformer   → Islam et al. 2024 (ECGformer, arXiv 2401.05434, >98%);
                  PatchECG (s41598, ViT sobre Chapman-Shaoxing)
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
import seaborn as sns

# =======================================================================
# 0. RUTAS Y CONSTANTES
# =======================================================================

PREPROCESSED_DIR = Path(
    r"C:\Users\Lina Castañeda\Documents\Originals\CHAPMAN-SHAOXING\preprocessed"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CLASS_NAMES = ["AFIB", "GSVT", "SB", "SR"]
N_CLASSES   = 4
N_LEADS     = 12
SIGNAL_LEN  = 5000   # muestras a 500 Hz


# =======================================================================
# 1. DATASET Y DATALOADERS
# =======================================================================

class ShaoxingDataset(Dataset):
    """
    Dataset PyTorch para Chapman-Shaoxing.
    Lee los arrays .npy generados en la Etapa 2.

    __getitem__ devuelve:
      signal : (12, 5000) float32  — transpuesta de (5000, 12)
               Las Conv1d de PyTorch esperan (batch, canales, longitud)
      label  : int64 escalar       — índice de clase (0-3)
    """
    def __init__(self, split: str):
        X = np.load(PREPROCESSED_DIR / f"shaoxing_{split}_X.npy")
        y = np.load(PREPROCESSED_DIR / f"shaoxing_{split}_y.npy")
        # Transponer: (N, 5000, 12) → (N, 12, 5000)
        self.X = torch.from_numpy(X.transpose(0, 2, 1))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(batch_size: int = 64,
                      num_workers: int = 4) -> tuple:
    """
    Construye los tres DataLoaders (train/val/test) y carga los pesos
    de clase para CrossEntropyLoss.

    batch_size=64 por defecto:
      Con 5000 muestras × 12 derivaciones la señal es 5× más larga que
      en PTB-XL. Para no exceder los 8 GB VRAM de la RTX 5050, se reduce
      el batch de 128 (Bloque 1) a 64. Ajustar a 32 si hay OOM.
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

# ── 2A. BLOQUE RESIDUAL (idéntico al Bloque 1) ─────────────────────────

class ResidualBlock1D(nn.Module):
    """
    Bloque residual 1D — igual que en PTB-XL/Bloque 1.
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
    CNN 1D ResNet-style adaptada a señales de 5000 muestras (500 Hz).

    Diferencia clave respecto al Bloque 1 (1000 muestras a 100 Hz):
      Se añade un MaxPool adicional en la entrada (5000→1250) y los
      strides internos comprimen más, para que la secuencia final
      tenga dimensión manejable antes del GlobalAvgPool.

    Flujo de dimensiones (con batch_size implícito):
      (12, 5000) → input_conv → (64, 625)
                 → res_block1 → (128, 625)
                 → res_block2 → (128, 313)   stride=2
                 → res_block3 → (256, 313)
                 → res_block4 → (256, 157)   stride=2
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
            nn.MaxPool1d(kernel_size=4, stride=4),   # 5000 → 1250 (era 2 en B1)
            nn.MaxPool1d(kernel_size=2, stride=2),   # 1250 → 625  (extra vs B1)
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

    Arquitectura:
      · Bloque CNN frontal: extrae características locales de morfología
        (ondas P, QRS, T) y reduce la longitud temporal a algo manejable
        para la LSTM (5000→156 muestras)
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
        # Objetivo: extraer características y reducir 5000 → ~156
        # para que la LSTM no tenga que procesar 5000 pasos de tiempo
        self.cnn = nn.Sequential(
            # Capa 1: kernel grande captura morfología completa
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),          # 5000 → 1250

            # Capa 2
            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),          # 1250 → 312

            # Capa 3
            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),          # 312 → 156

            nn.Dropout(dropout),
        )
        # Salida CNN: (batch, 128, 156)

        # ── LSTM bidireccional ────────────────────────────────────────
        # input_size=128 (canales de la CNN)
        # La LSTM recibe la secuencia transpuesta: (batch, 156, 128)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # Salida LSTM: (batch, 156, lstm_hidden*2) — bidireccional

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
        # x: (batch, 12, 5000)
        x = self.cnn(x)                    # (batch, 128, 156)
        x = x.permute(0, 2, 1)            # (batch, 156, 128) para LSTM
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

    Idea clave: en lugar de procesar los 5000 pasos uno a uno,
    la señal se divide en PATCHES (ventanas solapadas), cada patch
    se proyecta linealmente como un token, y el Transformer aprende
    relaciones entre patches mediante self-attention.

    Esto es análogo a cómo ViT trata imágenes (patches 2D),
    pero aquí son patches 1D de señal ECG.

    Flujo:
      (batch, 12, 5000)
        → CNN stem: reduce a (batch, 128, 156) y extrae features locales
        → Patchify: cada posición temporal es un token → (batch, 156, 128)
        → Positional encoding: inyecta posición en la secuencia
        → N capas Transformer Encoder (self-attention + FFN)
        → CLS token o GlobalAvgPool → (batch, d_model)
        → Clasificador FC → (batch, 4)

    Referencia: Islam et al. 2024 (ECGformer, arXiv 2401.05434)
                PatchECG (s41598026415495)
    """
    def __init__(self, n_leads=12, n_classes=N_CLASSES,
                 d_model=128, n_heads=8, n_layers=4,
                 dim_ff=256, dropout=0.1):
        super().__init__()

        # ── CNN stem (igual que en CNN_LSTM: reduce 5000→156) ─────────
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(64, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, stride=4),

            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2),
        )
        # Salida stem: (batch, d_model=128, 156)
        # Longitud de secuencia para el Transformer: seq_len = 156

        # ── CLS token ─────────────────────────────────────────────────
        # Token de clasificación aprendible (como en BERT/ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── Positional encoding (aprendible) ──────────────────────────
        # +1 para el CLS token
        self.pos_embedding = nn.Parameter(torch.zeros(1, 157, d_model))

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
        # x: (batch, 12, 5000)
        B = x.size(0)

        x = self.cnn_stem(x)              # (B, 128, 156)
        x = x.permute(0, 2, 1)           # (B, 156, 128)

        # Añadir CLS token al inicio de la secuencia
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 128)
        x = torch.cat([cls, x], dim=1)           # (B, 157, 128)

        # Positional encoding
        x = self.dropout(x + self.pos_embedding)

        # Transformer encoder
        x = self.transformer(x)           # (B, 157, 128)

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"best_{model_name}.pt"

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
    ckpt   = torch.load(OUTPUT_DIR / f"best_{model_name}.pt",
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


def plot_training_curves(model_name: str, history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val")
    axes[0].set_title("Pérdida (CrossEntropyLoss)")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Accuracy (val)",
                 color="tab:blue")
    axes[1].plot(epochs, history["val_f1"],  label="F1 macro (val)",
                 color="tab:orange")
    axes[1].set_title("Métricas de validación")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1); axes[1].legend()

    fig.suptitle(f"{model_name} — curvas de entrenamiento (Chapman 500 Hz)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{model_name}_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model_name: str, test_metrics: dict):
    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {model_name} (test)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{model_name}_confusion.png", dpi=150)
    plt.close(fig)


def save_test_report(model_name: str, test_metrics: dict):
    numeric = {k: v for k, v in test_metrics.items()
               if k not in ("classification_report", "confusion_matrix")}
    with open(OUTPUT_DIR / f"{model_name}_test_metrics.json",
              "w", encoding="utf-8") as f:
        json.dump(numeric, f, indent=2)

    lines = [
        "=" * 60,
        f"RESULTADOS TEST — {model_name} (Chapman-Shaoxing 500 Hz)",
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
    (OUTPUT_DIR / f"{model_name}_test_metrics.txt").write_text(
        "\n".join(lines), encoding="utf-8")


# =======================================================================
# 8. MAIN
# =======================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ──────────────────────────────────────────────────
    # batch_size=64: señal 5× más larga que PTB-XL → mitad del batch
    # num_workers=4: misma configuración que Bloque 1 (Ryzen 5 5600X)
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        batch_size=64, num_workers=4
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
        model = cfg["class"](**cfg["kwargs"])

        # ── Entrenamiento ─────────────────────────────────────────────
        _, history, ckpt_path = train_model(
            model_name=cfg["name"],
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            n_epochs=50,
            lr=1e-3,
        )

        # ── Guardar historial y curvas ────────────────────────────────
        save_history_csv(history,
                         OUTPUT_DIR / f"{cfg['name']}_history.csv")
        plot_training_curves(cfg["name"], history)

        # ── Evaluación en test ────────────────────────────────────────
        test_metrics = evaluate_on_test(
            model_name=cfg["name"],
            model_class=cfg["class"],
            model_kwargs=cfg["kwargs"],
            test_loader=test_loader,
            class_weights=class_weights,
        )

        # ── Guardar resultados ────────────────────────────────────────
        save_test_report(cfg["name"], test_metrics)
        plot_confusion_matrix(cfg["name"], test_metrics)
        all_test_results[cfg["name"]] = test_metrics

    # ── Tabla comparativa final ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARATIVA FINAL — CHAPMAN-SHAOXING 500 Hz")
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

    print(f"\n✓ Etapa 3 del Bloque 2 completada.")
    print(f"  Outputs en: {OUTPUT_DIR}")