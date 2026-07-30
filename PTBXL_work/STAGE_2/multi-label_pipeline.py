"""
=======================================================================
BLOQUE 1 - ETAPA 2: Pipeline de preprocesamiento PTB-XL
=======================================================================
Tarea    : Clasificación multi-label (NORM | MI | STTC | CD | HYP)
Entrada  : Archivos .dat/.hea de PTB-XL + ptbxl_database.csv
Salida   : Tensores listos para entrenar + pesos de clase
Literatura de referencia:
  - Wagner et al. 2020 (creadores de PTB-XL, definición de folds y superclases)
  - Hannun et al. 2022 (revisión sistemática, benchmark 100 Hz)
  - TransECG (filtro Butterworth 0.5-40 Hz, z-score, padding)
=======================================================================
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset, DataLoader

# ── CONFIGURACIÓN GLOBAL ──────────────────────────────────────────────
# Ajusta PTB_XL_PATH al directorio raíz donde descargaste PTB-XL
PTB_XL_PATH   = r"C:\Users\Lina Castañeda\Documents\Originals\PTBXL-RAW"
SAMPLING_RATE  = 100        # Hz (usa la carpeta records100)
SIGNAL_LENGTH  = 1000       # muestras = 10 s × 100 Hz
N_LEADS        = 12         # derivaciones estándar
N_CLASSES      = 5          # NORM, MI, STTC, CD, HYP
SUPERCLASSES   = ["NORM", "MI", "STTC", "CD", "HYP"]

# Folds según la convención oficial PTB-XL
FOLD_TEST      = 10         # fold reservado como test (Wagner et al. 2020)
FOLD_VAL       = 9          # fold para validación
# Folds 1-8 se usan para entrenamiento

# ── 1. CARGA DEL METADATA ─────────────────────────────────────────────

def load_metadata(ptb_xl_path: str) -> pd.DataFrame:
    """
    Carga ptbxl_database.csv y decodifica las columnas SCP que vienen
    como strings de diccionarios Python (e.g. "{'NORM': 100.0}").
    Excluye los 411 registros sin etiqueta diagnóstica.
    """
    csv_path = os.path.join(ptb_xl_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col="ecg_id")

    # Decodificar scp_codes de string a dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # Nota: las columnas "age" y "sex" se preservan tal cual en el DataFrame
    # para uso posterior (información clínica opcional en PTBXLDataset)

    # Cargar tabla de códigos SCP para identificar superclases diagnósticas
    scp_path = os.path.join(ptb_xl_path, "scp_statements.csv")
    scp_df   = pd.read_csv(scp_path, index_col=0)

    # Filtrar solo los códigos que tienen superclase diagnóstica (diagnostic=1)
    diagnostic_codes = scp_df[scp_df["diagnostic"] == 1]

    # Construir mapa: código SCP → superclase (e.g. "AFIB" → "MI")
    code_to_superclass = {}
    for code, row in diagnostic_codes.iterrows():
        sc = row["diagnostic_class"]
        if pd.notna(sc):
            code_to_superclass[code] = sc

    # ── Generar vector de etiquetas binario por registro ──────────────
    def build_label_vector(scp_dict: dict) -> list:
        """
        Devuelve [NORM, MI, STTC, CD, HYP] como lista de 0s y 1s.
        Solo cuenta un código si tiene confianza ≥ 50% (convención Wagner).
        """
        active = set()
        for code, confidence in scp_dict.items():
            if confidence >= 50 and code in code_to_superclass:
                active.add(code_to_superclass[code])
        return [1 if sc in active else 0 for sc in SUPERCLASSES]

    df["label_vector"] = df["scp_codes"].apply(build_label_vector)

    # ── Excluir registros sin ninguna superclase diagnóstica ──────────
    df["n_labels"] = df["label_vector"].apply(sum)
    n_before = len(df)
    df = df[df["n_labels"] > 0].copy()
    n_after  = len(df)
    print(f"Registros cargados  : {n_before}")
    print(f"Excluidos (sin dx)  : {n_before - n_after}")
    print(f"Registros útiles    : {n_after}")

    return df


# ── 2. FILTRO PASO-BANDA BUTTERWORTH ──────────────────────────────────

def bandpass_filter(signal: np.ndarray,
                    lowcut: float  = 0.5,
                    highcut: float = 40.0,
                    fs: int        = 100,
                    order: int     = 3) -> np.ndarray:
    """
    Aplica un filtro Butterworth paso-banda de forma cero (filtfilt).
    signal : array (n_muestras, n_derivaciones)
    Retorna: array de la misma forma filtrado.

    Por qué Butterworth orden 3:
      - Respuesta plana en la banda de paso (sin ripple)
      - filtfilt aplica el filtro dos veces (ida y vuelta) → fase cero,
        no distorsiona la morfología de las ondas P, QRS, T
    """
    nyq = fs / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # Filtrar cada derivación por separado
    return filtfilt(b, a, signal, axis=0)


# ── 3. NORMALIZACIÓN Z-SCORE POR DERIVACIÓN ───────────────────────────

def zscore_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalización z-score independiente por derivación.
    signal : array (n_muestras, n_derivaciones)
    Retorna: array normalizado (media≈0, std≈1 por canal)

    Por qué z-score en lugar de min-max:
      - Los ECG pueden tener picos artefactuales extremos (movimiento de
        electrodos). Min-max comprimiría toda la señal útil hacia cero.
      - Z-score es robusto a outliers y es la norma en la literatura de
        clasificación de ECG clínico (Wagner 2020, Hannun 2022).
    """
    mean = signal.mean(axis=0, keepdims=True)   # (1, 12)
    std  = signal.std(axis=0, keepdims=True)     # (1, 12)
    return (signal - mean) / (std + eps)


# ── 4. CARGA Y PREPROCESAMIENTO DE UNA SEÑAL ──────────────────────────

def load_and_preprocess(record_path: str,
                        ptb_xl_path: str,
                        sampling_rate: int = 100) -> np.ndarray:
    """
    Lee un registro .dat/.hea con wfdb, aplica filtro y normalización.
    Retorna tensor numpy (1000, 12) → float32.

    Padding/truncado: aunque PTB-XL garantiza exactamente 10 s,
    añadimos manejo defensivo para compatibilidad con otros datasets
    (útil cuando repliquemos el pipeline en Shaoxing).
    """
    # record_path en ptbxl_database.csv ya incluye la subcarpeta (records100/... o records500/...)
    full_path = os.path.join(ptb_xl_path, record_path)

    record = wfdb.rdrecord(full_path)
    signal = record.p_signal.astype(np.float32)  # (1000, 12)

    # Reemplazar NaN por cero (dato corrupto puntual)
    signal = np.nan_to_num(signal, nan=0.0)

    # Filtrado paso-banda
    signal = bandpass_filter(signal, fs=sampling_rate).astype(np.float32)

    # Ajustar longitud a SIGNAL_LENGTH (padding con ceros o truncado)
    n = signal.shape[0]
    if n < SIGNAL_LENGTH:
        pad = np.zeros((SIGNAL_LENGTH - n, N_LEADS), dtype=np.float32)
        signal = np.vstack([signal, pad])
    else:
        signal = signal[:SIGNAL_LENGTH]

    # Normalización z-score
    signal = zscore_normalize(signal).astype(np.float32)

    return signal  # (1000, 12)


# ── 5. DATASET DE PYTORCH ─────────────────────────────────────────────

class PTBXLDataset(Dataset):
    """
    Dataset de PyTorch para PTB-XL multi-label.

    __getitem__ devuelve:
      signal : tensor float32  (12, 1000)  — canales primero (Conv1D estándar)
      label  : tensor float32  (5,)        — vector binario [NORM,MI,STTC,CD,HYP]
    """

    def __init__(self,
                 df: pd.DataFrame,
                 ptb_xl_path: str,
                 sampling_rate: int = 100,
                 augment: bool = False,
                 use_clinical: bool = False):
        """
        df           : subset del DataFrame (train / val / test)
        ptb_xl_path  : ruta raíz de PTB-XL
        augment      : si True, aplica augmentación ligera (solo en train)
        use_clinical : si True, devuelve también un tensor con age y sex
        """
        self.df            = df.reset_index()
        self.ptb_xl_path   = ptb_xl_path
        self.sampling_rate = sampling_rate
        self.augment       = augment
        self.use_clinical  = use_clinical

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Ruta relativa guardada en ptbxl_database.csv (sin extensión)
        record_path = row["filename_lr"] if self.sampling_rate == 100 \
                      else row["filename_hr"]

        signal = load_and_preprocess(
            record_path, self.ptb_xl_path, self.sampling_rate
        )  # (1000, 12)

        # Augmentación ligera solo en entrenamiento
        if self.augment:
            signal = self._augment(signal)

        # Conv1D en PyTorch espera (canales, longitud) → transponer
        signal_tensor = torch.tensor(signal.T, dtype=torch.float32)  # (12, 1000)

        label = np.array(row["label_vector"], dtype=np.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)      # (5,)

        # Información clínica opcional: age (normalizada /100) y sex (0.0=Male, 1.0=Female)
        if self.use_clinical:
            age = np.float32(row["age"]) / 100.0
            sex = np.float32(row["sex"])  # PTB-XL: 0=Male, 1=Female
            clinical_tensor = torch.tensor([age, sex], dtype=torch.float32)  # (2,)
            return signal_tensor, label_tensor, clinical_tensor

        return signal_tensor, label_tensor

    @staticmethod
    def _augment(signal: np.ndarray) -> np.ndarray:
        """
        Augmentación conservadora para señales ECG:
          1. Ruido gaussiano leve (SNR alto para no distorsionar morfología)
          2. Escala de amplitud aleatoria ±10%
        NO se usa inversión temporal ni desplazamiento de fase porque
        alterarían la polaridad clínica de las derivaciones.
        """
        # Ruido aditivo gaussiano
        noise = np.random.normal(0, 0.01, signal.shape).astype(np.float32)
        signal = signal + noise

        # Escala de amplitud
        scale = np.random.uniform(0.9, 1.1)
        signal = signal * scale

        return signal


# ── 6. FUNCIÓN PRINCIPAL: CONSTRUIR DATALOADERS ───────────────────────

def build_dataloaders(ptb_xl_path: str,
                      batch_size: int   = 64,
                      num_workers: int  = 2,
                      sampling_rate: int = 100,
                      use_clinical: bool = False):
    """
    Carga metadata, divide por folds oficiales y devuelve tres DataLoaders.

    Retorna:
      train_loader, val_loader, test_loader, pos_weights
      pos_weights : tensor (5,) para BCEWithLogitsLoss
    """
    print("=" * 60)
    print("ETAPA 2 — Pipeline de preprocesamiento PTB-XL")
    print("=" * 60)

    # ── Cargar y filtrar metadata ──────────────────────────────────────
    df = load_metadata(ptb_xl_path)

    # ── División por strat_fold (sin mezclar pacientes) ───────────────
    df_test  = df[df["strat_fold"] == FOLD_TEST]
    df_val   = df[df["strat_fold"] == FOLD_VAL]
    df_train = df[~df["strat_fold"].isin([FOLD_TEST, FOLD_VAL])]

    print(f"\nDivisión de registros:")
    print(f"  Train (folds 1-8) : {len(df_train):>5} registros")
    print(f"  Val   (fold  9)   : {len(df_val):>5} registros")
    print(f"  Test  (fold 10)   : {len(df_test):>5} registros")

    # ── Calcular pos_weights para BCEWithLogitsLoss ───────────────────
    # pos_weight[i] = (N - N_pos_i) / N_pos_i   (solo sobre train)
    train_labels = np.stack(df_train["label_vector"].values)  # (N_train, 5)
    n_train      = len(df_train)
    n_pos        = train_labels.sum(axis=0)                   # (5,)
    n_neg        = n_train - n_pos
    pos_weights  = torch.tensor(n_neg / (n_pos + 1e-8), dtype=torch.float32)

    print(f"\nPos_weights para BCEWithLogitsLoss:")
    for cls, w in zip(SUPERCLASSES, pos_weights.tolist()):
        print(f"  {cls:<6} : {w:.4f}")

    # ── Construir Datasets ────────────────────────────────────────────
    train_ds = PTBXLDataset(df_train, ptb_xl_path, sampling_rate, augment=True,  use_clinical=use_clinical)
    val_ds   = PTBXLDataset(df_val,   ptb_xl_path, sampling_rate, augment=False, use_clinical=use_clinical)
    test_ds  = PTBXLDataset(df_test,  ptb_xl_path, sampling_rate, augment=False, use_clinical=use_clinical)

    # ── Construir DataLoaders ─────────────────────────────────────────
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"\nBatch size : {batch_size}")
    print(f"Batches train : {len(train_loader)}")
    print(f"Batches val   : {len(val_loader)}")
    print(f"Batches test  : {len(test_loader)}")
    print("=" * 60)
    print("Pipeline listo. Tensores de entrada: (batch, 12, 1000)")
    print("Tensores de salida (label): (batch, 5)")
    if use_clinical:
        print("Clínico activo: cada batch devuelve un tercer tensor (batch, 2) [age, sex]")
    print("=" * 60)

    return train_loader, val_loader, test_loader, pos_weights


# ── 7. VERIFICACIÓN RÁPIDA ────────────────────────────────────────────

def sanity_check(train_loader, pos_weights):
    """
    Carga un batch y verifica formas, rango de valores y etiquetas.
    Corre esto antes de entrenar para detectar errores de configuración.
    """
    print("\n── Sanity check ──")
    batch = next(iter(train_loader))
    if len(batch) == 3:
        signals, labels, clinical = batch
    else:
        signals, labels = batch
        clinical = None

    print(f"Shape señales : {signals.shape}")   # esperado: (64, 12, 1000)
    print(f"Shape labels  : {labels.shape}")    # esperado: (64, 5)
    print(f"Rango señal   : [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"Dtype señal   : {signals.dtype}")
    print(f"Dtype label   : {labels.dtype}")
    print(f"Pos_weights   : {pos_weights.tolist()}")

    if clinical is not None:
        print(f"Shape clínico : {clinical.shape}")   # esperado: (64, 2)
        print(f"Rango clínico : [{clinical.min():.3f}, {clinical.max():.3f}]")

    # Verificar que no hay NaN en el batch
    assert not torch.isnan(signals).any(), "¡NaN detectado en señales!"
    assert not torch.isnan(labels).any(),  "¡NaN detectado en labels!"
    print("✓ Sin NaN en el primer batch")

    # Mostrar un ejemplo de label
    print(f"\nEjemplo label (registro 0):")
    for cls, val in zip(SUPERCLASSES, labels[0].tolist()):
        estado = "✓" if val == 1.0 else "·"
        print(f"  {estado} {cls}")


# ── PUNTO DE ENTRADA ──────────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, pos_weights = build_dataloaders(
        ptb_xl_path   = PTB_XL_PATH,
        batch_size    = 64,
        num_workers   = 2,
        sampling_rate = SAMPLING_RATE,
    )
    sanity_check(train_loader, pos_weights)