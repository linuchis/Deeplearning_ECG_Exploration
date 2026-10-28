"""
=======================================================================
BLOQUE 1 - ETAPA 3 - MODELO 3 (V2): ECG Transformer regularizado
=======================================================================
Arquitectura : Transformer encoder con patch embeddings sobre señal 1D
Tarea        : Clasificación multi-label ECG (NORM|MI|STTC|CD|HYP)
Entrada      : (batch, 12, 1000)  — 12 derivaciones × 1000 muestras
Salida       : (batch, 5)         — logits (sin sigmoid, para BCEWithLogitsLoss)

Idea central (de arXiv:2401.05434 — ECGformer):
  "We split a series of waves into patches and provide the sequence
   of linear embeddings of these patches as an input to a Transformer.
   Signal patches are treated the same way as tokens (words) in NLP."

Diferencia fundamental respecto a CNN y CNN-LSTM:
  - CNN     : kernels locales → ve un segmento a la vez
  - CNN-LSTM: CNN local + LSTM secuencial → ve segmentos en orden
  - Transformer: atención global → cada parche ve TODOS los demás
                 simultáneamente, sin restricción de localidad ni orden

Por qué esto importa para PTB-XL:
  El paper micromachines1401155 señala que la atención estándar tiene
  "receptive field amplio que puede incluir información irrelevante".
  Nuestra implementación añade un extractor CNN liviano antes del
  Transformer (igual que CNN-DVIT) para dar contexto local antes de
  la atención global — balance entre localidad y largo alcance.

=======================================================================
QUÉ CAMBIA EN ESTA V2 (respecto a STAGE_3/TRANSFORMER/transformer-v1.py)
=======================================================================
La V1 (4.03M parámetros) quedó por debajo del CNN-ResNet (AUC 0.8910
vs 0.9185, F1 macro 0.6622 vs 0.7129) — la hipótesis de partida es que
con ~16k registros de train, la capacidad de atención global del
Transformer sobreajusta antes de converger del todo. Tres cambios,
los tres puramente de regularización/capacidad — SIN tocar arquitectura,
learning rate, scheduler ni presupuesto de épocas — para poder atribuir
cualquier diferencia de resultado específicamente a esto:

  1. dropout: 0.1 → 0.3
     Se aplica dentro de la atención (MultiheadAttention) y del FFN de
     cada bloque encoder. Subirlo fuerza a que ninguna cabeza de
     atención pueda depender de un subconjunto fijo de tokens —
     obliga a representaciones más distribuidas y robustas. Es la
     palanca de regularización más directa disponible en la arquitectura.

  2. weight_decay: 1e-2 → 5e-2
     AdamW penaliza pesos grandes en las capas lineales del FFN
     (d_model × mlp_ratio = 256×4 = 1024 unidades internas, la parte
     con más parámetros del modelo). Con 4M de parámetros y 16K
     muestras de entrenamiento, esos pesos crecen sin restricción
     suficiente bajo 1e-2. Quintuplicar el decay es agresivo pero
     proporcional a la magnitud del sobreajuste observado en la V1.

  3. n_layers: 4 → 2
     El cambio más impactante: reduce los parámetros de ~4.03M a
     ~2.1M, más comparable con el CNN-ResNet (1.72M). Con 2 capas,
     cada token solo "pregunta" dos veces a los demás antes de llegar
     al clasificador — limita la capacidad de memorización sin
     eliminar la ventaja central del Transformer (atención global).

Qué esperar: el sobreajuste debería comenzar más tarde que en la V1
(que divergía temprano); el AUC máximo probablemente no supere al
CNN-ResNet (0.9185), pero un resultado entre 0.90-0.915 demostraría que,
con regularización adecuada, el Transformer se acerca al baseline
convolucional — el checkpoint guardado sería más representativo de la
capacidad real de la arquitectura que el de la V1.

Referencias:
  - Islam et al. 2024 (arXiv:2401.05434) — ECGformer
  - micromachines1401155 — CNN-DVIT con deformable attention
  - Vaswani et al. 2017 — Attention Is All You Need (arquitectura base)
=======================================================================
"""

import torch
# cuDNN autotuning: igual que en los Modelos 1 y 2. La entrada tiene forma
# fija (12, 1000) en todos los batches, así que cuDNN puede fijarse en el
# algoritmo de convolución más rápido para la CNN local del PatchEmbedding
# sin pagar el costo de re-buscar en cada forward.
torch.backends.cudnn.benchmark = True        # autotuning de kernels CUDA
torch.backends.cudnn.deterministic = False   # más velocidad, menos reproducibilidad exacta
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report, multilabel_confusion_matrix
import time
import math
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
# los 5 modelos de STAGE_3 (consistencia visual entre modelos). Validada
# con el validador de paletas categoricas (banda de luminosidad, piso de
# croma, separacion bajo daltonismo protan/deutan, piso de vision normal).
PALETTE = ["darkorchid", "seagreen", "royalblue", "darkorange", "firebrick"]
CLASS_COLORS = dict(zip(SUPERCLASSES, PALETTE))
# Colormap secuencial para las matrices de confusion (blanco -> royalblue),
# en vez de un colormap generico tipo "Blues"/"viridis".
CM_CMAP = LinearSegmentedColormap.from_list("ptbxl_cm", ["#ffffff", "royalblue"])

MODEL_NAME = "ECGTransformer (V2 — regularizado)"

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
# A nivel de módulo (no dentro de "if __name__ == '__main__':") porque
# DataLoader con num_workers>0 usa multiprocessing "spawn" en Windows:
# cada worker relanza este archivo y necesita poder reimportar
# "multi-label_pipeline" para deserializar PTBXLDataset. El bug original
# en este archivo (sys.path.append(".") + "from ptbxl_pipeline import
# build_dataloaders") apuntaba a un módulo que nunca existió — el archivo
# real es STAGE_2/multi-label_pipeline.py.
import sys
import importlib

# .parent.parent.parent porque este archivo vive en STAGE_3/TRANSFORMER-V2/,
# misma profundidad que el resto de carpetas de modelos en STAGE_3
# (todos dos niveles bajo la raíz del proyecto).
_STAGE2_DIR = Path(__file__).resolve().parent.parent.parent / "STAGE_2"
if str(_STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(_STAGE2_DIR))

_ptbxl_pipeline = importlib.import_module("multi-label_pipeline")
build_dataloaders = _ptbxl_pipeline.build_dataloaders
PTB_XL_PATH = _ptbxl_pipeline.PTB_XL_PATH


# ── 1. PATCH EMBEDDING ────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Divide la señal ECG en parches y los proyecta a un espacio de
    dimensión d_model mediante una convolución 1D (equivalente a una
    proyección lineal por parche, más eficiente).

    Analogía con NLP:
      - Señal ECG (1000 muestras) → "oración"
      - Parche de P muestras      → "palabra" (token)
      - Proyección lineal         → "embedding de la palabra"

    Con patch_size=50 y stride=50 (sin solapamiento):
      1000 / 50 = 20 parches → secuencia de 20 tokens
      Cada token representa 500 ms de señal (a 100 Hz)
      Cubre aproximadamente 1 latido cardíaco completo

    La CNN liviana previa (kernel 15) extrae contexto local antes de
    que el Transformer aplique atención global. Esto replica la idea
    del CNN-DVIT: "CNN backbone for local feature extraction + Transformer
    for global dependencies".
    """

    def __init__(self,
                 n_leads:    int = 12,
                 patch_size: int = 50,
                 d_model:    int = 256,
                 dropout:    float = 0.1):
        super().__init__()

        self.patch_size = patch_size

        # CNN liviana: extrae contexto local antes del parcheo
        # Mantiene la longitud (padding=7 con kernel=15)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),   # GELU es la activación estándar en Transformers
        )

        # Proyección de parche → token de dimensión d_model
        # Conv1d con kernel=patch_size y stride=patch_size es equivalente
        # a una proyección lineal aplicada a cada parche sin solapamiento
        self.patch_proj = nn.Conv1d(
            64, d_model, kernel_size=patch_size, stride=patch_size, bias=False
        )

        # LayerNorm por token, fuera de patch_proj a propósito: Conv1d entrega
        # (batch, d_model, n_patches) con d_model en la dimensión 1, y
        # nn.LayerNorm(d_model) normaliza sobre la ÚLTIMA dimensión del tensor.
        # Aplicarlo ahí (como estaba antes, dentro del mismo Sequential que
        # el Conv1d) normalizaba sobre n_patches en vez de d_model y rompía
        # con RuntimeError: "expected input with shape [*, 256], but got
        # [4, 256, 20]". Por eso se aplica aquí, después del permute.
        self.patch_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, 12, 1000)
        out : (batch, n_patches, d_model)
              n_patches = 1000 / patch_size = 20
        """
        x = self.local_cnn(x)        # (batch, 64, 1000)
        x = self.patch_proj(x)       # (batch, d_model, n_patches)
        x = x.permute(0, 2, 1)      # (batch, n_patches, d_model)

        # Ahora sí: d_model es la última dimensión, como espera LayerNorm
        x = self.patch_norm(x)
        return self.dropout(x)


# ── 2. CODIFICACIÓN POSICIONAL ────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Añade información de posición a cada token.

    El Transformer por sí solo no sabe el orden de los tokens — trata
    la secuencia como un conjunto (set), no como una secuencia ordenada.
    La codificación posicional inyecta esa información.

    Usamos codificación posicional APRENDIDA (no sinusoidal fija) porque:
      - Con solo 20 tokens (parches), el número de parámetros es mínimo
        (20 × d_model = 5120 parámetros extra)
      - Puede adaptarse a los patrones específicos de PTB-XL
      - Es la opción usada en ViT (Vision Transformer) y sus derivados
        para señales biomédicas

    Forma: (1, max_patches, d_model) — el 1 permite broadcasting sobre el batch
    """

    def __init__(self, max_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_patches, d_model)"""
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])


# ── 3. BLOQUE TRANSFORMER ENCODER ────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Un bloque del encoder Transformer estándar (Vaswani et al. 2017).

    Estructura:
      x → LayerNorm → MultiHeadAttention → + residual
        → LayerNorm → FFN (MLP) → + residual

    Pre-LayerNorm (en lugar del post-LayerNorm original):
      La normalización antes de la atención (Pre-LN) es más estable
      durante el entrenamiento y no requiere warm-up del learning rate.
      Es la variante usada en GPT-2, ViT y la mayoría de Transformers modernos.

    FFN (Feed-Forward Network):
      Dos capas lineales con GELU y dropout.
      La dimensión interna es típicamente 4×d_model (mlp_ratio=4).
      Introduce no-linealidad y capacidad de transformación por token.
    """

    def __init__(self,
                 d_model:   int,
                 n_heads:   int,
                 mlp_ratio: int   = 4,
                 dropout:   float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = n_heads,
            dropout      = dropout,
            batch_first  = True,   # (batch, seq, features)
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_patches, d_model)

        La atención multi-cabeza calcula:
          Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) · V
        donde Q=K=V=x (self-attention): cada token se compara con todos
        los demás tokens para determinar cuánta atención prestarle.
        """
        # Rama de atención (Pre-LN + residual)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Rama FFN (Pre-LN + residual)
        x = x + self.ffn(self.norm2(x))

        return x


# ── 4. MODELO COMPLETO: ECG TRANSFORMER ──────────────────────────────

class ECGTransformer(nn.Module):
    """
    Transformer encoder para clasificación multi-label de ECG.

    Flujo completo:
      (batch, 12, 1000)
        ↓  PatchEmbedding (CNN local + proyección)
      (batch, 20, 256)      ← 20 tokens de dim 256
        ↓  PositionalEncoding
      (batch, 20, 256)      ← tokens con información de posición
        ↓  [CLS token] prepend
      (batch, 21, 256)      ← token de clasificación + 20 tokens ECG
        ↓  N × TransformerEncoderBlock
      (batch, 21, 256)
        ↓  Extraer CLS token [posición 0]
      (batch, 256)
        ↓  encoder_norm (LayerNorm) + encoder_proj (Linear→GELU→Dropout)
      (batch, proj_dim)     ← embedding latente (reutilizable, ver get_embeddings)
        ↓  [opcional: concat variables clínicas] + head (Linear)
      (batch, n_classes)    ← logits

    El token [CLS] (classification token):
      - Es un vector aprendible inicializado en ceros
      - Se prepende a la secuencia de parches
      - Después de N bloques de atención, el CLS ha "visto" todos los
        parches (porque la atención es global) y acumula la representación
        global del ECG completo
      - Se usa como entrada al clasificador (igual que en BERT y ViT)
      - Ventaja sobre pooling: el modelo aprende QUÉ información del ECG
        es relevante para la clasificación, no simplemente promedia todo

    Hiperparámetros elegidos (V2 — ver docstring del módulo para el porqué):
      patch_size=50 → 20 parches (1 latido aprox. por parche a 60 bpm)
      d_model=256   → balanceo entre capacidad y parámetros
      n_heads=8     → 8 cabezas de atención (d_model/n_heads = 32 por cabeza)
      n_layers=2    → V1 usaba 4; se reduce a 2 para bajar de ~4.03M a
                      ~2.1M parámetros y limitar el sobreajuste observado
      mlp_ratio=4   → FFN de dimensión 1024 internamente
      dropout=0.3   → V1 usaba 0.1; se sube para forzar representaciones
                      de atención más distribuidas (ver docstring del módulo)
    """

    def __init__(self,
                 n_leads:      int   = 12,
                 n_classes:    int   = 5,
                 patch_size:   int   = 50,
                 d_model:      int   = 256,
                 n_heads:      int   = 8,
                 n_layers:     int   = 4,
                 mlp_ratio:    int   = 4,
                 dropout:      float = 0.1,
                 proj_dim:     int   = 128,
                 use_clinical: bool  = False):
        super().__init__()

        self.use_clinical = use_clinical

        n_patches = 1000 // patch_size   # 20

        # Embedding de parches
        self.patch_embed = PatchEmbedding(
            n_leads=n_leads, patch_size=patch_size,
            d_model=d_model, dropout=dropout
        )

        # Token [CLS] aprendible
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Codificación posicional (+1 por el CLS token)
        self.pos_enc = PositionalEncoding(
            max_patches=n_patches + 1, d_model=d_model, dropout=dropout
        )

        # Bloques encoder
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                d_model=d_model, n_heads=n_heads,
                mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Encoder: normalización del CLS token + proyección al espacio latente
        # (proj_dim), reutilizable como embedding para transfer learning.
        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cabeza de clasificación: opcionalmente concatena variables clínicas
        # (p. ej. edad, sexo) al embedding antes de proyectar a n_classes.
        self.head = nn.Linear(proj_dim + (2 if use_clinical else 0), n_classes)

        self._init_weights()

    def _init_weights(self):
        """Inicialización estándar para Transformers (trunc_normal)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, clinical=None):
        """
        x        : (batch, 12, 1000)
        clinical : (batch, 2) opcional — solo se usa si use_clinical=True
        retorna  : (batch, n_classes) — logits
        """
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.encoder_norm(x[:, 0, :])
        z = self.encoder_proj(cls_out)
        if self.use_clinical and clinical is not None:
            z = torch.cat([z, clinical], dim=1)
        return self.head(z)

    def get_embeddings(self, x):
        """Devuelve el vector latente (batch, proj_dim) sin pasar por la cabeza."""
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.encoder_norm(x[:, 0, :])
        return self.encoder_proj(cls_out)


# ── MÉTRICAS ──────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_probs, threshold=0.5):
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
    for cls, f1 in zip(SUPERCLASSES,
                       f1_score(all_labels, all_preds,
                                average=None, zero_division=0)):
        metrics[f"f1_{cls}"] = float(f1)

    # Reporte completo (precision/recall/f1/support por clase) para el registro en outputs/
    metrics["classification_report"] = classification_report(
        all_labels, all_preds, target_names=SUPERCLASSES, zero_division=0
    )
    return metrics


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Entrena una época. Cada batch del loader puede traer 2 elementos
    (signals, labels) o 3 (signals, clinical, labels) — este último caso
    solo se da cuando el modelo se instancia con use_clinical=True."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        if len(batch) == 3:
            signals, clinical, labels = batch
            signals, clinical, labels = signals.to(device), clinical.to(device), labels.to(device)
            logits = model(signals, clinical)
        else:
            signals, labels = batch
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
        optimizer.zero_grad()
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()   # OneCycleLR: paso por batch, no por época
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_raw: bool = False):
    """Evalúa el modelo. Cada batch del loader puede traer 2 elementos
    (signals, labels) o 3 (signals, clinical, labels) — este último caso
    solo se da cuando el modelo se instancia con use_clinical=True."""
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for batch in loader:
        if len(batch) == 3:
            signals, clinical, labels = batch
            signals, clinical, labels = signals.to(device), clinical.to(device), labels.to(device)
            logits = model(signals, clinical)
        else:
            signals, labels = batch
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
        total_loss += criterion(logits, labels).item()
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


def train_model(train_loader, val_loader, pos_weights,
                n_epochs=50, lr=3e-4, save_path="best_transformer_v2.pt"):
    """
    Entrenamiento del Transformer.

    Diferencias respecto a los modelos anteriores:
      - lr=3e-4: learning rate estándar para Transformers (Karpathy rule)
      - OneCycleLR en lugar de ReduceLROnPlateau:
          Los Transformers se benefician de un warm-up gradual del LR
          seguido de un decay. OneCycleLR implementa esto automáticamente:
          sube el LR en los primeros ~30% de los pasos y luego lo baja.
          Esto estabiliza el entrenamiento en las primeras épocas donde
          la atención aún no ha convergido.
      - weight_decay=5e-2 (V1 usaba 1e-2): mayor regularización L2 porque
          los Transformers tienen más parámetros en las capas lineales
          del FFN. Se quintuplicó respecto a la V1 porque el modelo
          sobreajustaba con 1e-2 — ver docstring del módulo para el
          diagnóstico completo (AUC/F1 de la V1 por debajo del CNN-ResNet).

    Cambios V2 respecto a V1 (los tres, deliberadamente aislados de
    lr/scheduler/n_epochs para poder atribuir el resultado únicamente
    a regularización + capacidad):
      dropout   0.1 → 0.3   (en la instanciación del modelo, abajo)
      n_layers    4 → 2     (en la instanciación del modelo, abajo)
      weight_decay 1e-2 → 5e-2  (en el optimizer, abajo)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo : {device}")

    model     = ECGTransformer(
        n_leads=12, n_classes=N_CLASSES,
        patch_size=50, d_model=256,
        n_heads=8,
        n_layers=2,        # V2: era 4 — ~4.03M → ~2.1M parámetros
        mlp_ratio=4,
        dropout=0.3        # V2: era 0.1 — fuerza atención más distribuida
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    # V2: weight_decay era 1e-2 en la V1; se quintuplica porque el FFN
    # (la parte con más parámetros del modelo) sobreajustaba con esa
    # penalización L2 — ver docstring del módulo.
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)

    # OneCycleLR: warm-up + cosine decay por batch
    total_steps = n_epochs * len(train_loader)
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,        # 10% de pasos para warm-up
        anneal_strategy="cos",
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros  : {n_params:,}")
    print(f"Épocas máx. : {n_epochs}")
    print("-" * 60)

    best_auc, patience_cnt, patience_max = 0.0, 0, 15
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_auc, val_f1 = val_metrics["auc_macro"], val_metrics["f1_macro"]
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


# ── EVALUACIÓN FINAL ──────────────────────────────────────────────────

def evaluate_on_test(model_path, test_loader, pos_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Debe coincidir EXACTAMENTE con la instanciación en train_model
    # (n_layers en particular determina cuántos TransformerEncoderBlock
    # tiene el state_dict guardado); si no coincide, load_state_dict falla.
    model  = ECGTransformer(
        n_leads=12, n_classes=N_CLASSES,
        patch_size=50, d_model=256,
        n_heads=8,
        n_layers=2,        # V2: era 4
        mlp_ratio=4,
        dropout=0.3        # V2: era 0.1 (no afecta shapes, pero se mantiene igual por consistencia)
    ).to(device)
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
# Mismo patrón que los Modelos 1 y 2, para que las curvas y reportes de
# los tres modelos sean directamente comparables.

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
    checkpoint_path = OUTPUT_DIR / "best_transformer_v2.pt"

    # ── Ajustes para Ryzen 5 5600X + GPU 8 GB VRAM (RTX 5050) ──────────
    # Mismo batch_size/num_workers que los Modelos 1 y 2. El Transformer
    # opera sobre 21 tokens de 256-d (mucho más corto que los 125 pasos
    # de la LSTM), así que su huella de VRAM por muestra es MENOR que la
    # del CNN-LSTM pese a tener atención global — la atención es O(n²)
    # en función del número de tokens, pero n=21 es pequeño. Si sale
    # "CUDA out of memory" de todas formas, bájalo a 64.
    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        ptb_xl_path=PTB_XL_PATH,  # misma ruta que en la Etapa 2
        batch_size=128,
        num_workers=4,
    )

    # n_epochs=50 con early stopping (patience=15, ya definido en train_model)
    # es el mismo presupuesto que los Modelos 1 y 2 — el riesgo real para
    # el Transformer no es quedarse corto de épocas sino sobreajustar antes
    # de la 15 por el tamaño del dataset (~16k registros de train) frente
    # a su capacidad de atención global; si eso ocurre, la señal es subir
    # dropout/weight_decay, no cambiar la arquitectura (ver docstring de
    # train_model: ya usa weight_decay=1e-2, más alto que los Modelos 1 y 2).
    model, history = train_model(
        train_loader, val_loader, pos_weights,
        n_epochs=50,
        lr=3e-4,
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