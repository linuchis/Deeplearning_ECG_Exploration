"""
=======================================================================
TRANSFER LEARNING CRUZADO: PTB-XL → Chapman-Shaoxing (100 Hz)
=======================================================================
Pregunta científica: ¿cuánto transfieren representaciones ECG aprendidas
en una tarea de MORFOLOGÍA (PTB-XL: NORM/MI/STTC/CD/HYP, multi-label,
población europea) a una tarea de RITMO (Chapman-Shaoxing: AFIB/GSVT/
SB/SR, multiclase, población china)? El objetivo NO es superar el
baseline entrenado desde cero en Chapman, sino cuantificar la brecha.

Método (linear probing clásico):
  1. Cargar cada encoder PTB-XL ya entrenado (CNN1D_ResNet, CNN_LSTM,
     ECGTransformer) desde su checkpoint y CONGELARLO por completo
     (requires_grad=False, modo eval). Las arquitecturas no se tocan.
  2. Cargar los datos de Chapman 100 Hz (ya preprocesados a 900
     muestras / 9 s, ver preprocessed_100hz/).
  3. Para cada encoder: extraer embeddings de 128-d una sola vez (el
     encoder es determinista en eval() y nunca se actualiza, así que
     recalcular el forward en cada época sería trabajo redundante) y
     entrenar SOLO una cabeza nueva nn.Linear(128, 4) sobre esos
     embeddings congelados.
  4. Evaluar la cabeza entrenada en el test set de Chapman.
  5. Comparar contra el baseline de Chapman entrenado desde cero.

Los tres encoders vienen de PTBXL_work/STAGE_3/ (checkpoints
entrenados con use_clinical=False — best_cnn1d.pt, best_cnn_lstm_v2.pt,
best_transformer_v2.pt). Las clases de arquitectura se redefinen aquí
tal cual (sin modificarlas) para que el archivo sea autocontenido.

Compatible con Windows / CPU / GPU (CUDA opcional, se detecta sola).
=======================================================================
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

# ── Rutas del proyecto ──────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Checkpoints de los encoders PTB-XL (STAGE_3, entrenados sin variables
# clínicas — use_clinical=False, head Linear(128, 5)). Solo se usa
# get_embeddings(): el head original de 5 clases morfológicas se descarta.
PTBXL_CHECKPOINTS = {
    "CNN1D_ResNet":   ROOT_DIR / "PTBXL_work" / "STAGE_3" / "CNN_1D"        / "outputs" / "best_cnn1d.pt",
    "CNN_LSTM":       ROOT_DIR / "PTBXL_work" / "STAGE_3" / "CNN-LSTM-V2"  / "outputs" / "best_cnn_lstm_v2.pt",
    "ECGTransformer": ROOT_DIR / "PTBXL_work" / "STAGE_3" / "TRANSFORMER-V2" / "outputs" / "best_transformer_v2.pt",
}

# Datos Chapman-Shaoxing 100 Hz ya preprocesados (arrays .npy listos)
PREPROCESSED_DIR = Path(r"C:\Users\Lina Castañeda\Documents\Originals\CHAPMAN-SHAOXING\preprocessed_100hz")

RHYTHM_CLASSES   = ["AFIB", "GSVT", "SB", "SR"]   # índices 0..3, fijo
N_RHYTHM_CLASSES = len(RHYTHM_CLASSES)
EMBED_DIM        = 128   # proj_dim de los tres encoders PTB-XL

# Baseline Chapman 100Hz entrenado desde cero (para la tabla comparativa
# final). Solo CNN1D_ResNet tiene resultado real hasta el momento; los
# otros dos quedan como "pendiente" hasta correr esos experimentos.
BASELINE_CHAPMAN_FROM_SCRATCH = {
    "CNN1D_ResNet":   {"accuracy": 0.9518, "f1_macro": 0.9469},
    "CNN_LSTM":       {"accuracy": 0.9513692162417375, "f1_macro": 0.9460026499787794},
    "ECGTransformer": {"accuracy": 0.9480642115203022, "f1_macro": 0.9432226533722377},
}


# ═══════════════════════════════════════════════════════════════════
# 1. ARQUITECTURAS PTB-XL (copiadas tal cual de PTBXL_work/STAGE_3;
#    NO se modifican — solo se instancian para cargar los checkpoints
#    y usar get_embeddings()).
# ═══════════════════════════════════════════════════════════════════

# ── Modelo 1: CNN1D_ResNet ────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 kernel_size:  int   = 7,
                 stride:       int   = 1,
                 dropout:      float = 0.2):
        super().__init__()
        padding = kernel_size // 2

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
        return self.relu(out + residual)


class CNN1D_ResNet(nn.Module):
    def __init__(self,
                 n_leads:     int   = 12,
                 n_classes:   int   = 5,
                 dropout:     float = 0.3,
                 proj_dim:    int   = 128,
                 use_clinical: bool = False):
        super().__init__()
        self.use_clinical = use_clinical

        self.encoder = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            ResidualBlock1D(64,  128, kernel_size=7, stride=1, dropout=dropout),
            ResidualBlock1D(128, 128, kernel_size=7, stride=2, dropout=dropout),
            ResidualBlock1D(128, 256, kernel_size=5, stride=1, dropout=dropout),
            ResidualBlock1D(256, 256, kernel_size=5, stride=2, dropout=dropout),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(proj_dim + (2 if use_clinical else 0), n_classes)

    def forward(self, x: torch.Tensor, clinical: torch.Tensor = None) -> torch.Tensor:
        z = self.encoder(x)
        if self.use_clinical and clinical is not None:
            z = torch.cat([z, clinical], dim=1)
        return self.head(z)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── Modelo 2: CNN_LSTM (V2 — max-pool temporal) ───────────────────────

class ConvFeatureExtractor(nn.Module):
    def __init__(self, n_leads: int = 12, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = x.permute(0, 2, 1)
        return x


class CNN_LSTM(nn.Module):
    def __init__(self,
                 n_leads:     int   = 12,
                 n_classes:   int   = 5,
                 hidden_size: int   = 256,
                 num_layers:  int   = 2,
                 dropout:     float = 0.3,
                 proj_dim:    int   = 128,
                 use_clinical: bool = False):
        super().__init__()
        self.use_clinical = use_clinical

        self.encoder_cnn = ConvFeatureExtractor(n_leads=n_leads, dropout=dropout)

        self.encoder_lstm = nn.LSTM(
            input_size   = 256,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        self.encoder_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(proj_dim + (2 if use_clinical else 0), n_classes)

    def forward(self, x: torch.Tensor, clinical: torch.Tensor = None) -> torch.Tensor:
        z = self.encoder_cnn(x)
        lstm_out, _ = self.encoder_lstm(z)
        pooled = lstm_out.max(dim=1).values
        z = self.encoder_proj(pooled)
        if self.use_clinical and clinical is not None:
            z = torch.cat([z, clinical], dim=1)
        return self.head(z)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_cnn(x)
        lstm_out, _ = self.encoder_lstm(z)
        pooled = lstm_out.max(dim=1).values
        return self.encoder_proj(pooled)


# ── Modelo 3: ECGTransformer (V2 — regularizado) ──────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self,
                 n_leads:    int = 12,
                 patch_size: int = 50,
                 d_model:    int = 256,
                 dropout:    float = 0.1):
        super().__init__()
        self.patch_size = patch_size

        self.local_cnn = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.patch_proj = nn.Conv1d(
            64, d_model, kernel_size=patch_size, stride=patch_size, bias=False
        )

        self.patch_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_cnn(x)
        x = self.patch_proj(x)
        x = x.permute(0, 2, 1)
        x = self.patch_norm(x)
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    pos_embedding tiene tamaño fijo (max_patches=21, fijado en el
    entrenamiento con señales de 1000 muestras -> 20 parches + CLS).
    Con Chapman (900 muestras -> 18 parches + CLS = 19 tokens) solo se
    usa un subconjunto de las posiciones aprendidas; no hay error porque
    x.size(1)=19 <= max_patches=21.
    """

    def __init__(self, max_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pos_embedding[:, :x.size(1), :])


class TransformerEncoderBlock(nn.Module):
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
            batch_first  = True,
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
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ECGTransformer(nn.Module):
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

        n_patches = 1000 // patch_size   # 20 — fijado por el entrenamiento original

        self.patch_embed = PatchEmbedding(
            n_leads=n_leads, patch_size=patch_size,
            d_model=d_model, dropout=dropout
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = PositionalEncoding(
            max_patches=n_patches + 1, d_model=d_model, dropout=dropout
        )

        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(
                d_model=d_model, n_heads=n_heads,
                mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.encoder_norm = nn.LayerNorm(d_model)
        self.encoder_proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(proj_dim + (2 if use_clinical else 0), n_classes)

    def forward(self, x, clinical=None):
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
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_out = self.encoder_norm(x[:, 0, :])
        return self.encoder_proj(cls_out)


# ═══════════════════════════════════════════════════════════════════
# 2. DATOS CHAPMAN-SHAOXING 100 Hz
# ═══════════════════════════════════════════════════════════════════

class ChapmanDataset(Dataset):
    """
    Envuelve los arrays .npy de un split (train/val/test) de Chapman.
    X llega como (N, 900, 12) — se transpone a (N, 12, 900), el formato
    (batch, leads, tiempo) que esperan los tres encoders PTB-XL.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.ascontiguousarray(X.transpose(0, 2, 1))   # (N, 900, 12) -> (N, 12, 900)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_chapman_dataloaders(preprocessed_dir: Path, batch_size: int = 128):
    """
    Carga los 3 splits de Chapman 100Hz + los pesos de clase, y arma
    los DataLoaders. num_workers=0: los arrays ya están en memoria como
    tensores (nada de I/O ni preprocesamiento pesado por muestra), así
    que el paralelismo de multiprocessing no aporta y solo añade
    complejidad/overhead de spawn en Windows.
    """
    splits = {}
    for split in ("train", "val", "test"):
        X = np.load(preprocessed_dir / f"shaoxing_{split}_X.npy")
        y = np.load(preprocessed_dir / f"shaoxing_{split}_y.npy")
        splits[split] = ChapmanDataset(X, y)

    class_weights = np.load(preprocessed_dir / "shaoxing_class_weights.npy").astype(np.float32).flatten()

    train_loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(splits["val"],   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(splits["test"],  batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, torch.from_numpy(class_weights)


# ═══════════════════════════════════════════════════════════════════
# 3. CARGA DE ENCODERS PTB-XL (congelados)
# ═══════════════════════════════════════════════════════════════════

def build_ptbxl_model(model_name: str) -> nn.Module:
    """Instancia la arquitectura PTB-XL correspondiente, con los mismos
    hiperparámetros usados al entrenar los checkpoints de STAGE_3
    (use_clinical=False -> head Linear(128, 5))."""
    if model_name == "CNN1D_ResNet":
        return CNN1D_ResNet(n_leads=12, n_classes=5, dropout=0.3, use_clinical=False)
    if model_name == "CNN_LSTM":
        return CNN_LSTM(n_leads=12, n_classes=5, hidden_size=256, num_layers=2,
                         dropout=0.3, use_clinical=False)
    if model_name == "ECGTransformer":
        return ECGTransformer(n_leads=12, n_classes=5, patch_size=50, d_model=256,
                               n_heads=8, n_layers=2, mlp_ratio=4, dropout=0.3,
                               use_clinical=False)
    raise ValueError(f"Modelo desconocido: {model_name}")


def load_frozen_encoder(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Carga el checkpoint PTB-XL completo (encoder + head original de
    5 clases morfológicas) y congela TODOS los parámetros. El head
    original no se usa nunca: solo llamamos a get_embeddings()."""
    model = build_ptbxl_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    return model


# ═══════════════════════════════════════════════════════════════════
# 4. EXTRACCIÓN DE EMBEDDINGS + ENTRENAMIENTO DE LA CABEZA LINEAL
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(encoder: nn.Module, loader: DataLoader, device: torch.device):
    """Pasa todas las señales de un split por el encoder congelado
    (torch.no_grad(): el encoder nunca hace backward) y devuelve
    (embeddings, labels) como tensores en CPU.

    Se hace UNA sola vez por split/encoder en vez de en cada época de
    entrenamiento de la cabeza: el encoder está en eval() y congelado,
    así que su salida es determinista — recalcularla cada época sería
    trabajo idéntico repetido sin ningún beneficio."""
    encoder.eval()
    all_embeddings, all_labels = [], []

    for signals, labels in loader:
        signals = signals.to(device)
        z = encoder.get_embeddings(signals)
        all_embeddings.append(z.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def make_embedding_loaders(encoder: nn.Module,
                           train_loader: DataLoader,
                           val_loader:   DataLoader,
                           test_loader:  DataLoader,
                           device:       torch.device,
                           batch_size:   int = 128):
    """Extrae embeddings de los 3 splits y arma DataLoaders sobre esos
    embeddings (128-d) en vez de sobre las señales crudas — así el
    entrenamiento de la cabeza lineal es casi instantáneo."""
    train_emb, train_y = extract_embeddings(encoder, train_loader, device)
    val_emb,   val_y   = extract_embeddings(encoder, val_loader,   device)
    test_emb,  test_y  = extract_embeddings(encoder, test_loader,  device)

    train_loader_emb = DataLoader(TensorDataset(train_emb, train_y), batch_size=batch_size, shuffle=True)
    val_loader_emb   = DataLoader(TensorDataset(val_emb,   val_y),   batch_size=batch_size, shuffle=False)
    test_loader_emb  = DataLoader(TensorDataset(test_emb,  test_y),  batch_size=batch_size, shuffle=False)

    return train_loader_emb, val_loader_emb, test_loader_emb


def train_linear_head(train_loader_emb: DataLoader,
                      val_loader_emb:   DataLoader,
                      class_weights:    torch.Tensor,
                      device:           torch.device,
                      n_epochs:         int   = 50,
                      lr:               float = 1e-3,
                      patience:         int   = 10,
                      save_path:        Path  = None):
    """
    Entrena nn.Linear(128, 4) sobre los embeddings congelados de un
    encoder PTB-XL. CrossEntropyLoss ponderada con los pesos de clase
    de Chapman (desbalance AFIB/GSVT/SB/SR). Early stopping sobre F1
    macro de validación (patience=10), máximo n_epochs=50.
    """
    head = nn.Linear(EMBED_DIM, N_RHYTHM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(head.parameters(), lr=lr)

    best_f1       = -1.0
    patience_cnt  = 0
    best_state    = {k: v.clone() for k, v in head.state_dict().items()}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        head.train()
        train_loss = 0.0
        for emb, labels in train_loader_emb:
            emb, labels = emb.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = head(emb)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader_emb)

        # Validación
        head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for emb, labels in val_loader_emb:
                emb = emb.to(device)
                logits = head(emb)
                val_preds.append(logits.argmax(dim=1).cpu().numpy())
                val_labels.append(labels.numpy())
        val_preds  = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        print(f"  Época {epoch:02d}/{n_epochs} | train_loss={train_loss:.4f} | val_F1_macro={val_f1:.4f} | {time.time()-t0:.1f}s")

        if val_f1 > best_f1:
            best_f1      = val_f1
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in head.state_dict().items()}
            if save_path is not None:
                torch.save({"head_state": best_state, "val_f1_macro": val_f1, "epoch": epoch}, save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping en época {epoch} (sin mejora en {patience} épocas).")
                break

    head.load_state_dict(best_state)
    return head, best_f1


@torch.no_grad()
def evaluate_head(head: nn.Module, test_loader_emb: DataLoader, device: torch.device) -> dict:
    """Evalúa la cabeza entrenada en el test set de Chapman: accuracy,
    F1 macro y F1 por clase de ritmo (AFIB/GSVT/SB/SR)."""
    head.eval()
    all_preds, all_labels = [], []

    for emb, labels in test_loader_emb:
        emb = emb.to(device)
        logits = head(emb)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }
    f1_per_class = f1_score(all_labels, all_preds, average=None,
                            zero_division=0, labels=list(range(N_RHYTHM_CLASSES)))
    for cls_name, f1 in zip(RHYTHM_CLASSES, f1_per_class):
        metrics[f"f1_{cls_name}"] = float(f1)

    return metrics


# ═══════════════════════════════════════════════════════════════════
# 5. TABLA COMPARATIVA FINAL
# ═══════════════════════════════════════════════════════════════════

def print_comparison_table(transfer_results: dict) -> None:
    print("\n" + "=" * 70)
    print("Baseline Chapman 100Hz (desde cero):")
    for name in PTBXL_CHECKPOINTS:
        base = BASELINE_CHAPMAN_FROM_SCRATCH.get(name)
        if base is None:
            print(f"  {name:<15} → (pendiente)")
        else:
            print(f"  {name:<15} → Acc={base['accuracy']:.4f}  F1={base['f1_macro']:.4f}")

    print("\nTransfer PTB-XL → Chapman 100Hz (encoder congelado):")
    for name in PTBXL_CHECKPOINTS:
        r = transfer_results[name]
        print(f"  {name:<15} → Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}")
    print("=" * 70)


def save_results_json(transfer_results: dict, path: Path) -> None:
    payload = {
        "baseline_chapman_from_scratch": BASELINE_CHAPMAN_FROM_SCRATCH,
        "transfer_ptbxl_to_chapman":     transfer_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Paso 2: datos Chapman 100Hz (señales crudas, para extraer embeddings)
    train_loader, val_loader, test_loader, class_weights = build_chapman_dataloaders(
        PREPROCESSED_DIR, batch_size=128
    )
    print(f"Pesos de clase Chapman ({RHYTHM_CLASSES}): {class_weights.tolist()}")

    transfer_results = {}

    for model_name, checkpoint_path in PTBXL_CHECKPOINTS.items():
        print("\n" + "=" * 70)
        print(f"ENCODER: {model_name}")
        print("=" * 70)

        # Paso 1: cargar y congelar el encoder PTB-XL
        encoder = load_frozen_encoder(model_name, checkpoint_path, device)
        n_frozen = sum(p.numel() for p in encoder.parameters())
        print(f"Parámetros del encoder (todos congelados): {n_frozen:,}")

        # Paso 3a: extraer embeddings de 128-d (una sola vez, no_grad)
        train_loader_emb, val_loader_emb, test_loader_emb = make_embedding_loaders(
            encoder, train_loader, val_loader, test_loader, device, batch_size=128
        )

        # Paso 3b: entrenar SOLO la cabeza lineal nn.Linear(128, 4)
        head_save_path = OUTPUT_DIR / f"head_{model_name}.pt"
        head, best_val_f1 = train_linear_head(
            train_loader_emb, val_loader_emb, class_weights, device,
            n_epochs=50, lr=1e-3, patience=10, save_path=head_save_path,
        )
        print(f"Mejor F1 macro (val): {best_val_f1:.4f}  (cabeza guardada en {head_save_path})")

        # Paso 4: evaluación en test
        test_metrics = evaluate_head(head, test_loader_emb, device)
        test_metrics["best_val_f1_macro"] = float(best_val_f1)
        transfer_results[model_name] = test_metrics

        print(f"\nTEST -> Acc={test_metrics['accuracy']:.4f}  F1 macro={test_metrics['f1_macro']:.4f}")
        for cls in RHYTHM_CLASSES:
            print(f"  F1 {cls:<6}: {test_metrics[f'f1_{cls}']:.4f}")

    # Paso 5: tabla comparativa + guardado en outputs/transfer_results.json
    print_comparison_table(transfer_results)
    results_path = OUTPUT_DIR / "transfer_results.json"
    save_results_json(transfer_results, results_path)
    print(f"\nResultados guardados en: {results_path}")
