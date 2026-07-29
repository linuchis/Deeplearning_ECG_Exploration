"""
=======================================================================
BLOQUE 1 - ETAPA 3 - MODELO 1: CNN 1D ResNet-style
=======================================================================
Arquitectura : CNN 1D con bloques residuales (ResNet-style)
Tarea        : Clasificación multi-label ECG (NORM|MI|STTC|CD|HYP)
Entrada      : (batch, 12, 1000)  — 12 derivaciones × 1000 muestras
Salida       : (batch, 5)         — logits (sin sigmoid, para BCEWithLogitsLoss)

Por qué esta arquitectura como baseline:
  - Las CNN 1D con conexiones residuales son el modelo de referencia
    en benchmarks de PTB-XL (paperswithcode.com/sota/rhythm-on-ptb-xl)
  - Las conexiones residuales (skip connections) resuelven el problema
    del gradiente evanescente en redes profundas (He et al. 2016)
  - Operan directamente sobre la señal cruda 1D, sin transformaciones
    previas, lo que las hace robustas y reproducibles
  - Hannun et al. 2022 (revisión de 230 estudios) identificó ResNet-CNN
    como la arquitectura con mejores resultados generales en ECG

Referencia: Wagner et al. 2020 (PTB-XL benchmark)
=======================================================================
"""

import torch
torch.backends.cudnn.benchmark = True        # autotuning de kernels CUDA
torch.backends.cudnn.deterministic = False   # más velocidad, menos reproducibilidad exacta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import time
import os
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
N_CLASSES    = 5
OUTPUT_DIR   = Path(__file__).resolve().parent / "outputs"

# ── Import del pipeline de la Etapa 2 (STAGE_2/multi-label_pipeline.py) ─
# Debe hacerse a nivel de módulo (no dentro de "if __name__ == '__main__':")
# porque DataLoader con num_workers>0 usa multiprocessing con método "spawn"
# en Windows: cada worker relanza este archivo como módulo y necesita poder
# volver a importar "multi-label_pipeline" para deserializar PTBXLDataset.
# Si el import vive solo dentro del bloque __main__, los workers no lo ven
# y falla con "ModuleNotFoundError: No module named 'ptbxl_pipeline'".
import sys
import importlib

_STAGE2_DIR = Path(__file__).resolve().parent.parent.parent / "STAGE_2"
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))

_ptbxl_pipeline = importlib.import_module("multi-label_pipeline")
build_dataloaders = _ptbxl_pipeline.build_dataloaders
PTB_XL_PATH = _ptbxl_pipeline.PTB_XL_PATH


# ── BLOQUE RESIDUAL 1D ────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    """
    Bloque residual estándar adaptado a señales 1D.

    Estructura de cada bloque:
      Conv1D → BatchNorm → ReLU → Dropout → Conv1D → BatchNorm
      + conexión residual (skip connection)
      → ReLU

    La conexión residual permite que el gradiente fluya directamente
    hacia capas anteriores durante backpropagation, evitando que
    desaparezca en redes profundas (problema conocido como
    vanishing gradient).

    Si el número de canales cambia entre entrada y salida, se usa
    una convolución 1×1 para ajustar la dimensión del skip.
    """

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 kernel_size:  int   = 7,
                 stride:       int   = 1,
                 dropout:      float = 0.2):
        super().__init__()

        padding = kernel_size // 2   # padding "same" manual

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        # Skip connection: ajusta canales y/o longitud si cambia el stride
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out      = self.conv_block(x)
        return self.relu(out + residual)   # suma elemento a elemento


# ── MODELO COMPLETO ───────────────────────────────────────────────────

class CNN1D_ResNet(nn.Module):
    """
    Red CNN 1D ResNet-style para clasificación multi-label de ECG.

    Arquitectura (de entrada a salida):
      1. Capa de entrada: Conv1D grande (kernel 15) para capturar
         patrones de largo alcance en la señal (ondas completas P-QRS-T)
      2. 4 bloques residuales con aumento progresivo de canales y
         reducción de la longitud temporal (stride=2 en bloques 2 y 4)
      3. Global Average Pooling: colapsa la dimensión temporal
         → invarianza a la posición de los patrones en la señal
      4. Clasificador fully-connected con dropout

    Canales: 12 → 64 → 128 → 128 → 256 → 256 → GAP → 128 → 5
    Longitud: 1000 → 500 → 500 → 250 → 250 (después de strides)
    """

    def __init__(self,
                 n_leads:   int   = 12,
                 n_classes: int   = 5,
                 dropout:   float = 0.3):
        super().__init__()

        # ── Capa de entrada ────────────────────────────────────────────
        # Kernel grande (15) para capturar morfología completa de la onda
        self.input_conv = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 1000 → 500
        )

        # ── Bloques residuales ─────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            ResidualBlock1D(64,  128, kernel_size=7, stride=1, dropout=dropout),
            ResidualBlock1D(128, 128, kernel_size=7, stride=2, dropout=dropout),  # 500→250
            ResidualBlock1D(128, 256, kernel_size=5, stride=1, dropout=dropout),
            ResidualBlock1D(256, 256, kernel_size=5, stride=2, dropout=dropout),  # 250→125
        )

        # ── Cabeza de clasificación ────────────────────────────────────
        # Global Average Pooling: promedia a lo largo del tiempo
        # Resultado: (batch, 256) — independiente de la longitud de entrada
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
            # Sin sigmoid aquí: BCEWithLogitsLoss lo aplica internamente
            # (más estable numéricamente)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 12, 1000)
        retorna: (batch, 5) — logits
        """
        x = self.input_conv(x)     # (batch, 64,  500)
        x = self.res_blocks(x)     # (batch, 256, 125)
        x = self.global_avg_pool(x)  # (batch, 256, 1)
        x = self.classifier(x)     # (batch, 5)
        return x


# ── MÉTRICAS DE EVALUACIÓN ────────────────────────────────────────────

def compute_metrics(all_labels: np.ndarray,
                    all_probs:  np.ndarray,
                    threshold:  float = 0.5) -> dict:
    """
    Calcula las métricas estándar para clasificación multi-label:
      - AUC-ROC macro: métrica principal en benchmarks PTB-XL
      - F1-score macro: con umbral 0.5 sobre las probabilidades
      - F1 por clase: para identificar dónde falla el modelo

    all_labels : (N, 5) binario
    all_probs  : (N, 5) probabilidades [0,1] (después de sigmoid)
    """
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {}

    # AUC-ROC macro (requiere que haya positivos en cada clase)
    try:
        metrics["auc_macro"] = float(roc_auc_score(
            all_labels, all_probs, average="macro"
        ))
    except ValueError:
        metrics["auc_macro"] = float("nan")

    # F1-score macro
    metrics["f1_macro"] = float(f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    ))

    # F1 por clase
    f1_per_class = f1_score(
        all_labels, all_preds, average=None, zero_division=0
    )
    for cls, f1 in zip(SUPERCLASSES, f1_per_class):
        metrics[f"f1_{cls}"] = float(f1)

    # Reporte completo (precision/recall/f1/support por clase) para el registro en outputs/
    metrics["classification_report"] = classification_report(
        all_labels, all_preds, target_names=SUPERCLASSES, zero_division=0
    )

    return metrics


# ── BUCLE DE ENTRENAMIENTO ────────────────────────────────────────────

def train_one_epoch(model:      nn.Module,
                    loader:     DataLoader,
                    optimizer:  torch.optim.Optimizer,
                    criterion:  nn.Module,
                    device:     torch.device) -> float:
    """Entrena el modelo por una época. Retorna la pérdida promedio."""
    model.train()
    total_loss = 0.0

    for signals, labels in loader:
        signals = signals.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        logits = model(signals)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping: evita explosión de gradientes
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model:     nn.Module,
             loader:    DataLoader,
             criterion: nn.Module,
             device:    torch.device) -> tuple[float, dict]:
    """Evalúa el modelo en val o test. Retorna (pérdida, métricas)."""
    model.eval()
    total_loss  = 0.0
    all_labels  = []
    all_probs   = []

    for signals, labels in loader:
        signals = signals.to(device)
        labels  = labels.to(device)

        logits = model(signals)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        # Sigmoid para convertir logits en probabilidades
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    metrics = compute_metrics(all_labels, all_probs)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics


# ── ENTRENAMIENTO COMPLETO ────────────────────────────────────────────

def train_model(train_loader: DataLoader,
                val_loader:   DataLoader,
                pos_weights:  torch.Tensor,
                n_epochs:     int   = 50,
                lr:           float = 1e-3,
                save_path:    str   = "best_cnn1d.pt") -> nn.Module:
    """
    Entrena CNN1D_ResNet con:
      - BCEWithLogitsLoss ponderada (pos_weights de la Etapa 2)
      - Optimizador Adam con weight decay (regularización L2)
      - ReduceLROnPlateau: reduce lr si la val loss no mejora
      - Early stopping: detiene si val AUC no mejora en 10 épocas
      - Guarda el mejor modelo según AUC-ROC en validación
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo : {device}")

    model     = CNN1D_ResNet(n_leads=12, n_classes=N_CLASSES, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights.to(device)
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Conteo de parámetros entrenables
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros  : {n_params:,}")
    print(f"Épocas máx. : {n_epochs}")
    print("-" * 60)

    best_auc     = 0.0
    patience_cnt = 0
    patience_max = 10   # early stopping

    history = {
        "train_loss": [], "val_loss": [],
        "val_auc": [], "val_f1": []
    }

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss          = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device
        )

        val_auc = val_metrics["auc_macro"]
        val_f1  = val_metrics["f1_macro"]

        # Actualizar scheduler (imprime manualmente el cambio de LR,
        # ya que ReduceLROnPlateau ya no acepta verbose= en esta versión de PyTorch)
        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            print(f"  ↓ LR reducido: {lr_before:.2e} → {lr_after:.2e}")

        # Guardar historial
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        elapsed = time.time() - t0
        print(f"Época {epoch:02d}/{n_epochs} | "
              f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"AUC={val_auc:.4f} F1={val_f1:.4f} | "
              f"{elapsed:.1f}s")

        # Early stopping y guardado del mejor modelo
        if val_auc > best_auc:
            best_auc     = val_auc
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_auc":     val_auc,
                "val_f1":      val_f1,
                "history":     history,
            }, save_path)
            print(f"  ✓ Mejor modelo guardado (AUC={best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience_max:
                print(f"\nEarly stopping en época {epoch} "
                      f"(sin mejora en {patience_max} épocas).")
                break

    print(f"\nMejor AUC en validación: {best_auc:.4f}")
    return model, history


# ── EVALUACIÓN FINAL EN TEST ──────────────────────────────────────────

def evaluate_on_test(model_path:   str,
                     test_loader:  DataLoader,
                     pos_weights:  torch.Tensor):
    """
    Carga el mejor checkpoint y evalúa en el fold de test oficial (fold 10).
    Imprime el reporte completo por clase.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = CNN1D_ResNet(n_leads=12, n_classes=N_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("RESULTADOS FINALES — TEST (fold 10)")
    print("=" * 60)
    print(f"Loss test   : {test_loss:.4f}")
    print(f"AUC macro   : {test_metrics['auc_macro']:.4f}")
    print(f"F1 macro    : {test_metrics['f1_macro']:.4f}")
    print("\nF1 por clase:")
    for cls in SUPERCLASSES:
        print(f"  {cls:<6} : {test_metrics[f'f1_{cls}']:.4f}")
    print("=" * 60)

    test_metrics["loss"] = float(test_loss)
    return test_metrics


# ── GUARDADO DE RESULTADOS (outputs/) ─────────────────────────────────

def save_history_csv(history: dict, path: Path) -> None:
    """Guarda train_loss/val_loss/val_auc/val_f1 por época en un CSV."""
    keys = list(history.keys())
    n_epochs = len(history[keys[0]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(n_epochs):
            writer.writerow([i + 1] + [history[k][i] for k in keys])


def plot_training_curves(history: dict, path: Path) -> None:
    """Grafica loss (train/val) y AUC/F1 de validación por época."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Pérdida (BCEWithLogitsLoss)")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_auc"], label="AUC macro (val)", color="tab:green")
    axes[1].plot(epochs, history["val_f1"], label="F1 macro (val)", color="tab:orange")
    axes[1].set_title("Métricas de validación")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("CNN1D_ResNet — curvas de entrenamiento")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_f1_per_class(test_metrics: dict, path: Path) -> None:
    """Grafica el F1-score por superclase en el fold de test."""
    values = [test_metrics[f"f1_{cls}"] for cls in SUPERCLASSES]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SUPERCLASSES)))
    bars = ax.bar(SUPERCLASSES, values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1)
    ax.set_title("F1-score por clase — test (fold 10)")
    ax.set_ylabel("F1-score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_test_report(test_metrics: dict, json_path: Path, txt_path: Path) -> None:
    """Guarda las métricas de test en JSON (numéricas) y en TXT (legible, con reporte por clase)."""
    numeric_metrics = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(numeric_metrics, f, indent=2, ensure_ascii=False)

    lines = [
        "=" * 60,
        "RESULTADOS FINALES — TEST (fold 10)",
        "=" * 60,
        f"Loss test : {test_metrics['loss']:.4f}",
        f"AUC macro : {test_metrics['auc_macro']:.4f}",
        f"F1 macro  : {test_metrics['f1_macro']:.4f}",
        "",
        "F1 por clase:",
    ]
    for cls in SUPERCLASSES:
        lines.append(f"  {cls:<6} : {test_metrics[f'f1_{cls}']:.4f}")
    lines.append("")
    lines.append("Reporte de clasificación (sklearn, umbral=0.5):")
    lines.append(test_metrics["classification_report"])
    lines.append("=" * 60)

    txt_path.write_text("\n".join(lines), encoding="utf-8")


# ── PUNTO DE ENTRADA ──────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "best_cnn1d.pt"

    # ── Ajustes para Ryzen 5 5600X + GPU 8 GB VRAM (RTX 5050) ──────────
    # num_workers=4: usa 4 de los 6 cores para cargar datos en paralelo
    # mientras la GPU entrena; deja 2 cores libres.
    # batch_size=128: con 8 GB VRAM este modelo cabe holgado a 128.
    # Si sale "CUDA out of memory", bájalo a 64.
    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        ptb_xl_path=PTB_XL_PATH,  # misma ruta que en la Etapa 2
        batch_size=128,
        num_workers=4,
    )

    # Entrenar
    model, history = train_model(
        train_loader, val_loader, pos_weights,
        n_epochs=50,
        lr=1e-3,
        save_path=str(checkpoint_path),
    )

    save_history_csv(history, OUTPUT_DIR / "training_history.csv")
    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")

    # Evaluar en test
    test_metrics = evaluate_on_test(str(checkpoint_path), test_loader, pos_weights)

    save_test_report(test_metrics, OUTPUT_DIR / "test_metrics.json", OUTPUT_DIR / "test_metrics.txt")
    plot_f1_per_class(test_metrics, OUTPUT_DIR / "test_f1_per_class.png")

    print(f"\nResultados guardados en: {OUTPUT_DIR}")