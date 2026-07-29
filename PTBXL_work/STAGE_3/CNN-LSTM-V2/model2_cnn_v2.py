"""
=======================================================================
BLOQUE 1 - ETAPA 3 - MODELO 2 (V2): CNN-LSTM Híbrido con readout max-pool
=======================================================================
Arquitectura : CNN 1D (extractor de características) + LSTM (modelado temporal)
Tarea        : Clasificación multi-label ECG (NORM|MI|STTC|CD|HYP)
Entrada      : (batch, 12, 1000)  — 12 derivaciones × 1000 muestras
Salida       : (batch, 5)         — logits (sin sigmoid, para BCEWithLogitsLoss)

Por qué esta arquitectura como segundo modelo:
  - La CNN extrae características morfológicas locales (ondas P, QRS, T)
    convirtiéndolas en una secuencia de vectores de características
  - La LSTM recibe esa secuencia y modela las dependencias temporales
    entre latidos consecutivos (ritmo, intervalos RR, variabilidad)
  - La combinación captura lo que la CNN sola no puede: el CONTEXTO
    temporal de los patrones morfológicos
  - Sun et al. (sensors2406306) demuestran que CNN-LSTM supera a CNN
    sola en clasificación de arritmias (97.1% vs 96.5% accuracy)
  - La variante CNN-LSTM-SE (con mecanismo de atención) llega a 98.5%

Diferencia clave vs Modelo 1 (CNN ResNet):
  - CNN ResNet: aprende "qué patrones existen en la señal"
  - CNN-LSTM  : aprende "qué patrones existen Y en qué orden temporal"

=======================================================================
QUÉ CAMBIA EN ESTA V2 (respecto a STAGE_3/CNN-LSTM/model2_cnn.py)
=======================================================================
En la V1, el CNN-LSTM entrenado con 50 épocas quedó por debajo del
Modelo 1 (CNN ResNet) en las 5 clases: AUC macro 0.9089 vs 0.9185,
F1 macro 0.6907 vs 0.7129. Las curvas de validación de la V1 mostraban
un val_loss mucho más ruidoso que el del Modelo 1 (picos de hasta 0.92
frente a ~0.82), con el AUC estable pero el F1/loss oscilando cada vez
más tras la reducción de LR — el patrón típico de un "readout" frágil,
no necesariamente de sobreajuste puro por capacidad.

El sospechoso: la V1 usaba many-to-one clásico, tomando SOLO el último
paso temporal de la secuencia LSTM (`lstm_out[:, -1, :]`) como resumen
de los 125 pasos. Eso depende de que las puertas de la LSTM propaguen
sin pérdida TODA la secuencia hasta el final — un único punto de fallo.
El Modelo 1 (CNN ResNet), en cambio, usa Global Average Pooling: agrega
información de los 125 pasos por igual, por lo que ningún paso aislado
puede arruinar la predicción.

Cambio aplicado en V2: reemplazar el último-estado-oculto por MAX
POOLING sobre la dimensión temporal completa de `lstm_out`
(`lstm_out.max(dim=1).values`). Cada una de las 512 features (256
forward + 256 backward) toma su valor más alto a lo largo de los 125
pasos, en vez de depender únicamente del paso 125.

Por qué max pooling y no otro ajuste (justificación metodológica):
  - Aísla UNA variable. Si en vez de esto hubiéramos bajado hidden_size
    o subido dropout/weight_decay, y el modelo hubiera mejorado, no
    sabríamos si mejoró porque el sobreajuste por capacidad bajó o
    porque el readout dejó de ser el cuello de botella — dos hipótesis
    mezcladas en un solo experimento, sin forma de atribuir la causa.
    Aquí solo se toca el mecanismo de agregación temporal; arquitectura,
    hidden_size, dropout, lr y presupuesto de épocas quedan IDÉNTICOS
    a la V1, así que cualquier diferencia en el resultado final es
    atribuible específicamente al cambio de readout.
  - Tiene respaldo directo en la literatura del proyecto:
      · Sun et al. (sensors2406306) resuelven el mismo problema de
        readout frágil con un módulo Squeeze-and-Excitation (SE) tras
        la CNN: hacen Global Average Pooling sobre los canales para
        detectar qué características son globalmente relevantes y
        reponderan la representación en consecuencia.
      · El paper de micromachines citado en el proyecto menciona
        spatial pyramid pooling como mecanismo para capturar
        dependencias entre latidos sin depender de segmentación
        explícita del ciclo cardíaco.
      · Max pooling sobre la secuencia LSTM persigue el mismo objetivo
        que ambos mecanismos —que el readout no dependa de un único
        punto temporal, sino de la activación más fuerte observada en
        toda la secuencia— con la implementación más simple posible
        (sin parámetros nuevos, sin módulos adicionales).

Referencia: Sun et al. 2024 (sensors2406306), Wagner et al. 2020
=======================================================================
"""

import torch
# cuDNN autotuning: prueba varios algoritmos de convolución en las primeras
# iteraciones y se queda con el más rápido para el tamaño de entrada fijo
# que usamos (12, 1000). Igual que en el Modelo 1 (CNN1D_ResNet).
torch.backends.cudnn.benchmark = True        # autotuning de kernels CUDA
torch.backends.cudnn.deterministic = False   # más velocidad, menos reproducibilidad exacta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report, multilabel_confusion_matrix
import time
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
N_CLASSES    = 5
OUTPUT_DIR   = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR_ES = OUTPUT_DIR / "es"
OUTPUT_DIR_EN = OUTPUT_DIR / "en"

# ── Paleta fija del proyecto (no generica) ─────────────────────────────
# Una entrada por superclase, en el mismo orden en TODAS las figuras de
# los 5 modelos de STAGE_3 (consistencia visual entre modelos).
PALETTE = ["blueviolet", "mediumblue", "aquamarine", "lightcoral", "gold"]
CLASS_COLORS = dict(zip(SUPERCLASSES, PALETTE))
# Colormap secuencial para las matrices de confusion (blanco -> mediumblue),
# en vez de un colormap generico tipo "Blues"/"viridis".
CM_CMAP = LinearSegmentedColormap.from_list("ptbxl_cm", ["#ffffff", "mediumblue"])

MODEL_NAME = "CNN_LSTM (V2 — max-pool temporal)"

# ── Textos ES/EN para graficos (mismos datos, mismos ejes) ────────────
TEXTS = {
    "es": {
        "loss_title": "Pérdida (BCEWithLogitsLoss)",
        "epoch": "Época",
        "loss_ylabel": "Loss",
        "train": "train",
        "val": "val",
        "metrics_title": "Métricas de validación",
        "score_ylabel": "Score",
        "auc_label": "AUC macro (val)",
        "f1_label": "F1 macro (val)",
        "curves_suptitle": "{model} — curvas de entrenamiento",
        "f1_title": "F1-score por clase — test (fold 10)",
        "f1_ylabel": "F1-score",
        "cm_suptitle": "{model} — matrices de confusión por clase (test, fold 10, umbral=0.5)",
        "predicted": "Predicho",
        "actual": "Real",
        "neg": "Negativo",
        "pos": "Positivo",
    },
    "en": {
        "loss_title": "Loss (BCEWithLogitsLoss)",
        "epoch": "Epoch",
        "loss_ylabel": "Loss",
        "train": "train",
        "val": "val",
        "metrics_title": "Validation metrics",
        "score_ylabel": "Score",
        "auc_label": "AUC macro (val)",
        "f1_label": "F1 macro (val)",
        "curves_suptitle": "{model} — training curves",
        "f1_title": "F1-score per class — test (fold 10)",
        "f1_ylabel": "F1-score",
        "cm_suptitle": "{model} — per-class confusion matrices (test, fold 10, threshold=0.5)",
        "predicted": "Predicted",
        "actual": "Actual",
        "neg": "Negative",
        "pos": "Positive",
    },
}

# ── Import del pipeline de la Etapa 2 (STAGE_2/multi-label_pipeline.py) ─
# Debe hacerse a nivel de módulo (no dentro de "if __name__ == '__main__':")
# porque DataLoader con num_workers>0 usa multiprocessing con método "spawn"
# en Windows: cada worker relanza este archivo como módulo y necesita poder
# volver a importar "multi-label_pipeline" para deserializar PTBXLDataset.
# Si el import vive solo dentro del bloque __main__, los workers no lo ven
# y falla con "ModuleNotFoundError: No module named 'ptbxl_pipeline'"
# (el bug original: sys.path.append(".") + "from ptbxl_pipeline import ..."
# — ese módulo nunca existió, el archivo real es STAGE_2/multi-label_pipeline.py).
import sys
import importlib

# .parent.parent.parent porque este archivo vive en STAGE_3/CNN-LSTM-V2/,
# dos niveles por debajo de la raíz del proyecto (misma profundidad que
# STAGE_3/CNN_1D/model1_cnn1d.py y STAGE_3/CNN-LSTM/model2_cnn.py, por
# eso el mismo número de ".parent" funciona igual aquí).
_STAGE2_DIR = Path(__file__).resolve().parent.parent.parent / "STAGE_2"
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))

_ptbxl_pipeline = importlib.import_module("multi-label_pipeline")
build_dataloaders = _ptbxl_pipeline.build_dataloaders
PTB_XL_PATH = _ptbxl_pipeline.PTB_XL_PATH


# ── BLOQUE CONVOLUCIONAL (EXTRACTOR DE CARACTERÍSTICAS) ───────────────

class ConvFeatureExtractor(nn.Module):
    """
    Extrae características morfológicas locales de la señal ECG.

    Convierte la señal de entrada (batch, 12, 1000) en una secuencia
    de vectores de características (batch, T, features) donde T es
    la longitud temporal reducida.

    La LSTM recibirá esta secuencia como una "oración" donde cada
    "palabra" es el vector de características de un segmento temporal.

    Diseño:
      3 bloques CNN progresivos con BatchNorm y MaxPooling.
      Cada MaxPool divide la longitud temporal por 2.
      1000 → 500 → 250 → 125 muestras temporales.
      Los canales aumentan: 12 → 64 → 128 → 256.
    """

    def __init__(self, n_leads: int = 12, dropout: float = 0.2):
        super().__init__()

        self.blocks = nn.Sequential(
            # Bloque 1: captura morfología gruesa (ondas completas)
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 1000 → 500

            # Bloque 2: captura detalles finos (picos, transiciones)
            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 500 → 250

            # Bloque 3: representación de alto nivel por segmento
            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),   # 250 → 125
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, 12, 1000)
        out : (batch, 125, 256)  ← transpuesto para LSTM
              LSTM espera (batch, seq_len, features)
        """
        x = self.blocks(x)       # (batch, 256, 125)
        x = x.permute(0, 2, 1)  # (batch, 125, 256) ← seq_len, features
        return x


# ── MODELO COMPLETO CNN-LSTM ──────────────────────────────────────────

class CNN_LSTM(nn.Module):
    """
    Arquitectura híbrida CNN-LSTM para clasificación multi-label de ECG.

    Flujo de datos:
      (batch, 12, 1000)
        ↓  ConvFeatureExtractor
      (batch, 125, 256)    ← secuencia de 125 vectores de 256 features
        ↓  LSTM bidireccional (2 capas)
      (batch, 125, 512)    ← 256 forward + 256 backward
        ↓  Max pooling sobre el tiempo (readout V2, ver docstring del módulo)
      (batch, 512)
        ↓  Clasificador FC
      (batch, 5)           ← logits

    Por qué LSTM bidireccional:
      - La LSTM unidireccional solo ve el pasado: h_t depende de x_1..x_t
      - La bidireccional también ve el futuro: combina contexto pasado y futuro
      - En ECG clínico tiene sentido: un cambio en la onda T puede
        reinterpretarse al ver el QRS que viene después
      - Duplica el tamaño del estado oculto (hidden_size × 2)

    Por qué max pooling sobre el tiempo en vez de último estado oculto (V1):
      - En la V1 se usaba many-to-one clásico (`lstm_out[:, -1, :]`),
        que depende de que las puertas de la LSTM propaguen sin pérdida
        toda la secuencia hasta el último paso — un único punto de fallo
        que coincidió con un val_loss inestable en el entrenamiento real.
      - Max pooling toma, para cada una de las 512 features, el valor
        más alto observado en cualquiera de los 125 pasos temporales.
        Ya no importa si la información relevante aparece al principio,
        en medio o al final de la secuencia: el readout la captura igual.
      - Es la misma motivación que el módulo SE de Sun et al. (pooling
        global + reponderación) y el spatial pyramid pooling del paper
        de micromachines, pero sin parámetros adicionales.
    """

    def __init__(self,
                 n_leads:     int   = 12,
                 n_classes:   int   = 5,
                 hidden_size: int   = 256,
                 num_layers:  int   = 2,
                 dropout:     float = 0.3):
        super().__init__()

        # Extractor CNN
        self.cnn = ConvFeatureExtractor(n_leads=n_leads, dropout=dropout)

        # LSTM bidireccional
        # input_size = 256 (features de salida del CNN)
        # hidden_size = 256, bidireccional → salida efectiva = 512
        self.lstm = nn.LSTM(
            input_size   = 256,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,        # (batch, seq, features)
            bidirectional = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # Clasificador
        # Entrada: hidden_size * 2 (bidireccional)
        lstm_out_size = hidden_size * 2

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
            # Sin sigmoid: BCEWithLogitsLoss lo aplica internamente
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 12, 1000)
        retorna: (batch, 5) — logits
        """
        # Extracción CNN
        features = self.cnn(x)           # (batch, 125, 256)

        # LSTM — retorna (output, (h_n, c_n))
        # output : (batch, 125, 512)
        # h_n    : (num_layers*2, batch, 256)  ← ya no se usa (era el readout de la V1)
        lstm_out, _ = self.lstm(features)

        # ── Readout V2: max pooling sobre la dimensión temporal ─────────
        # V1 usaba solo lstm_out[:, -1, :] (último paso temporal), lo que
        # hacía que la predicción dependiera enteramente de que las puertas
        # de la LSTM hubieran propagado bien la información hasta el final
        # de la secuencia. Aquí, para cada una de las 512 features
        # (256 forward + 256 backward) se toma su activación máxima a lo
        # largo de los 125 pasos temporales — el patrón más fuerte "gana"
        # sin importar en qué paso ocurrió.
        # .values porque torch.max(dim=...) devuelve (valores, índices);
        # solo nos interesan los valores para el clasificador.
        pooled = lstm_out.max(dim=1).values   # (batch, 125, 512) → (batch, 512)

        logits = self.classifier(pooled)  # (batch, 5)
        return logits


# ── MÉTRICAS (idénticas al Modelo 1 para comparación justa) ──────────

def compute_metrics(all_labels: np.ndarray,
                    all_probs:  np.ndarray,
                    threshold:  float = 0.5) -> dict:
    all_preds = (all_probs >= threshold).astype(int)
    metrics   = {}
    try:
        metrics["auc_macro"] = float(roc_auc_score(
            all_labels, all_probs, average="macro"
        ))
    except ValueError:
        metrics["auc_macro"] = float("nan")

    metrics["f1_macro"] = float(f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    ))
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
def evaluate(model, loader, criterion, device, return_raw: bool = False):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        logits  = model(signals)
        loss    = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    metrics = compute_metrics(all_labels, all_probs)
    avg_loss = total_loss / len(loader)
    if return_raw:
        return avg_loss, metrics, all_labels, all_probs
    return avg_loss, metrics


def compute_confusion_matrices(all_labels: np.ndarray,
                               all_probs:  np.ndarray,
                               threshold:  float = 0.5) -> np.ndarray:
    """Matriz de confusión binaria por superclase (esquema multi-label).
    Retorna array (n_clases, 2, 2): [[TN, FP], [FN, TP]] por clase."""
    all_preds = (all_probs >= threshold).astype(int)
    return multilabel_confusion_matrix(all_labels, all_preds)


# ── ENTRENAMIENTO COMPLETO ────────────────────────────────────────────

def train_model(train_loader, val_loader, pos_weights,
                n_epochs=50, lr=5e-4, save_path="best_cnn_lstm_v2.pt"):
    """
    Misma lógica de entrenamiento que el Modelo 1 y que la V1 de este
    modelo (mismo lr, mismo optimizador, mismo early stopping) — el
    único cambio respecto a la V1 es el readout dentro de CNN_LSTM.forward
    (max pooling en vez de último estado oculto). Ver docstring del módulo.
    Nota: lr=5e-4 (en lugar de 1e-3) porque la LSTM es más sensible
    a learning rates altos — puede desestabilizarse con lr=1e-3.
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo : {device}")

    model     = CNN_LSTM(n_leads=12, n_classes=N_CLASSES,
                         hidden_size=256, num_layers=2, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # verbose= fue removido de ReduceLROnPlateau en esta versión de PyTorch
    # (mismo ajuste que en el Modelo 1); el cambio de LR se imprime a mano abajo.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros  : {n_params:,}")
    print(f"Épocas máx. : {n_epochs}")
    print("-" * 60)

    best_auc, patience_cnt, patience_max = 0.0, 0, 10
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    for epoch in range(1, n_epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_auc, val_f1 = val_metrics["auc_macro"], val_metrics["f1_macro"]

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after != lr_before:
            print(f"  ↓ LR reducido: {lr_before:.2e} → {lr_after:.2e}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        print(f"Época {epoch:02d}/{n_epochs} | "
              f"Loss train={train_loss:.4f} val={val_loss:.4f} | "
              f"AUC={val_auc:.4f} F1={val_f1:.4f} | "
              f"{time.time()-t0:.1f}s")

        if val_auc > best_auc:
            best_auc, patience_cnt = val_auc, 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_auc": val_auc, "val_f1": val_f1, "history": history,
            }, save_path)
            print(f"  ✓ Mejor modelo guardado (AUC={best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience_max:
                print(f"\nEarly stopping en época {epoch}.")
                break

    print(f"\nMejor AUC en validación: {best_auc:.4f}")
    return model, history


# ── EVALUACIÓN FINAL EN TEST ──────────────────────────────────────────

def evaluate_on_test(model_path, test_loader, pos_weights):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = CNN_LSTM(n_leads=12, n_classes=N_CLASSES,
                         hidden_size=256, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    test_loss, test_metrics, all_labels, all_probs = evaluate(
        model, test_loader, criterion, device, return_raw=True
    )

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
    cms = compute_confusion_matrices(all_labels, all_probs)
    return test_metrics, cms


# ── GUARDADO DE RESULTADOS (outputs/) ─────────────────────────────────
# Idéntico al Modelo 1, para que las curvas y reportes de ambos modelos
# sean directamente comparables (misma estructura de archivos en outputs/).

def save_history_csv(history: dict, path: Path) -> None:
    """Guarda train_loss/val_loss/val_auc/val_f1 por época en un CSV."""
    keys = list(history.keys())
    n_epochs = len(history[keys[0]])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(n_epochs):
            writer.writerow([i + 1] + [history[k][i] for k in keys])


def plot_training_curves(history: dict, path: Path, lang: str = "es") -> None:
    """Grafica loss (train/val) y AUC/F1 de validación por época."""
    t = TEXTS[lang]
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label=t["train"], color=PALETTE[0])
    axes[0].plot(epochs, history["val_loss"], label=t["val"], color=PALETTE[1])
    axes[0].set_title(t["loss_title"])
    axes[0].set_xlabel(t["epoch"])
    axes[0].set_ylabel(t["loss_ylabel"])
    axes[0].legend()

    axes[1].plot(epochs, history["val_auc"], label=t["auc_label"], color=PALETTE[3])
    axes[1].plot(epochs, history["val_f1"], label=t["f1_label"], color=PALETTE[4])
    axes[1].set_title(t["metrics_title"])
    axes[1].set_xlabel(t["epoch"])
    axes[1].set_ylabel(t["score_ylabel"])
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle(t["curves_suptitle"].format(model=MODEL_NAME))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_f1_per_class(test_metrics: dict, path: Path, lang: str = "es") -> None:
    """Grafica el F1-score por superclase en el fold de test."""
    t = TEXTS[lang]
    values = [test_metrics[f"f1_{cls}"] for cls in SUPERCLASSES]
    colors = [CLASS_COLORS[cls] for cls in SUPERCLASSES]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(SUPERCLASSES, values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1)
    ax.set_title(t["f1_title"])
    ax.set_ylabel(t["f1_ylabel"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(cms: np.ndarray, path: Path, lang: str = "es") -> None:
    """Grafica las 5 matrices de confusión binarias (una por superclase)."""
    t = TEXTS[lang]
    tick_labels = [t["neg"], t["pos"]]
    fig, axes = plt.subplots(1, len(SUPERCLASSES), figsize=(3.2 * len(SUPERCLASSES), 3.6))

    for ax, cls, cm in zip(axes, SUPERCLASSES, cms):
        vmax = cm.max() if cm.max() > 0 else 1
        ax.imshow(cm, cmap=CM_CMAP, vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                value = int(cm[i, j])
                color = "white" if value > vmax * 0.55 else "#2b2b2b"
                ax.text(j, i, value, ha="center", va="center", color=color, fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel(t["predicted"], fontsize=8)
        ax.set_ylabel(t["actual"], fontsize=8)
        ax.set_title(cls, color=CLASS_COLORS[cls], fontweight="bold")

    fig.suptitle(t["cm_suptitle"].format(model=MODEL_NAME))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_test_report(test_metrics: dict, json_path: Path,
                     txt_path_es: Path, txt_path_en: Path) -> None:
    """Guarda las métricas de test en JSON (numéricas, en la raíz de outputs/)
    y en TXT legible con reporte por clase, uno en es/ y otro en en/."""
    numeric_metrics = {k: v for k, v in test_metrics.items() if k != "classification_report"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(numeric_metrics, f, indent=2, ensure_ascii=False)

    lines_es = [
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
        lines_es.append(f"  {cls:<6} : {test_metrics[f'f1_{cls}']:.4f}")
    lines_es.append("")
    lines_es.append("Reporte de clasificación (sklearn, umbral=0.5):")
    lines_es.append(test_metrics["classification_report"])
    lines_es.append("=" * 60)
    txt_path_es.write_text("\n".join(lines_es), encoding="utf-8")

    lines_en = [
        "=" * 60,
        "FINAL RESULTS — TEST (fold 10)",
        "=" * 60,
        f"Test loss : {test_metrics['loss']:.4f}",
        f"AUC macro : {test_metrics['auc_macro']:.4f}",
        f"F1 macro  : {test_metrics['f1_macro']:.4f}",
        "",
        "F1 per class:",
    ]
    for cls in SUPERCLASSES:
        lines_en.append(f"  {cls:<6} : {test_metrics[f'f1_{cls}']:.4f}")
    lines_en.append("")
    lines_en.append("Classification report (sklearn, threshold=0.5):")
    lines_en.append(test_metrics["classification_report"])
    lines_en.append("=" * 60)
    txt_path_en.write_text("\n".join(lines_en), encoding="utf-8")


# ── PUNTO DE ENTRADA ──────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_ES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_EN.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "best_cnn_lstm_v2.pt"

    # ── Ajustes para Ryzen 5 5600X + GPU 8 GB VRAM (RTX 5050) ──────────
    # Mismos batch_size/num_workers que el Modelo 1: la LSTM añade cómputo
    # secuencial pero no cambia el tamaño de los tensores de entrada, así
    # que la huella de VRAM es comparable. Si sale "CUDA out of memory",
    # bájalo a 64 (la LSTM sí usa más memoria que la CNN pura por los
    # estados ocultos bidireccionales que mantiene por paso temporal).
    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        ptb_xl_path=PTB_XL_PATH,  # misma ruta que en la Etapa 2
        batch_size=128,
        num_workers=4,
    )

    model, history = train_model(
        train_loader, val_loader, pos_weights,
        n_epochs=50,
        lr=5e-4,      # más conservador que CNN sola (ver docstring de train_model)
        save_path=str(checkpoint_path),
    )

    save_history_csv(history, OUTPUT_DIR / "training_history.csv")
    plot_training_curves(history, OUTPUT_DIR_ES / "training_curves.png", lang="es")
    plot_training_curves(history, OUTPUT_DIR_EN / "training_curves.png", lang="en")

    test_metrics, cms = evaluate_on_test(str(checkpoint_path), test_loader, pos_weights)

    save_test_report(
        test_metrics, OUTPUT_DIR / "test_metrics.json",
        OUTPUT_DIR_ES / "test_metrics.txt", OUTPUT_DIR_EN / "test_metrics.txt",
    )
    plot_f1_per_class(test_metrics, OUTPUT_DIR_ES / "test_f1_per_class.png", lang="es")
    plot_f1_per_class(test_metrics, OUTPUT_DIR_EN / "test_f1_per_class.png", lang="en")
    plot_confusion_matrices(cms, OUTPUT_DIR_ES / "confusion_matrices.png", lang="es")
    plot_confusion_matrices(cms, OUTPUT_DIR_EN / "confusion_matrices.png", lang="en")

    print(f"\nResultados guardados en: {OUTPUT_DIR}")
    print(f"Graficos + reportes ES en: {OUTPUT_DIR_ES}")
    print(f"Graficos + reportes EN en: {OUTPUT_DIR_EN}")