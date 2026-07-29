# =============================================================================
# BLOQUE 2 — CHAPMAN-SHAOXING
# Etapa 1: Exploración del dataset
# Etapa 2: Preprocesamiento (pipeline definitivo)
#
# Decisiones metodológicas documentadas:
#   · 500 Hz nativos — sin resampleo (Zheng 2020, BimodalCNN)
#   · Filtro Butterworth paso-banda 0.5–40 Hz bidireccional
#   · Normalización Z-score por derivación + clip ±4σ (PatchECG; ver Etapa 2)
#   · División estratificada 70/10/20 con semilla fija (reproducible)
#   · CrossEntropyLoss + softmax — tarea multiclase (1 ritmo por paciente)
#
# Ajuste de prestaciones (mismo criterio que PTBXL_work, STAGE_3):
#   · Máquina: Ryzen 5 5600X (6 núcleos / 12 hilos) + RTX 5050 8 GB VRAM
#   · Chapman-Shaoxing tiene ~10.500 señales de 5000×12 muestras. Cargar y
#     filtrar (Butterworth bidireccional × 12 derivaciones) cada una en un
#     solo proceso es serial y no aprovecha la máquina.
#   · En PTBXL_work el paralelismo se aplica vía DataLoader(num_workers=4),
#     dejando 2 de los 6 núcleos libres para el hilo principal / E-S. Aquí no
#     hay DataLoader (el pipeline arma arrays .npy directamente), así que se
#     replica el mismo criterio con ProcessPoolExecutor(max_workers=4).
#
# Referencias:
#   Zheng et al. (2020) Scientific Data 7:48
#   Sensors 2024, 24, 7896 (knowledge distillation)
#   BimodalCNN (co-training grayscale + scalogram)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =============================================================================

BASE_DIR = r"C:\Users\Lina Castañeda\Documents\Originals\CHAPMAN-SHAOXING"

def _resolver_ruta(base_dir: str, nombre_base: str) -> str:
    """
    La colección de Figshare (Zheng 2020) distribuye Diagnostics y
    AttributesDictionary como .xlsx; algunos mirrors los traen como .csv.
    Se acepta cualquiera de las dos extensiones, priorizando .csv si existe.
    """
    ruta_csv  = os.path.join(base_dir, nombre_base + ".csv")
    ruta_xlsx = os.path.join(base_dir, nombre_base + ".xlsx")
    return ruta_csv if os.path.exists(ruta_csv) else ruta_xlsx

DIAGNOSTICS_PATH = _resolver_ruta(BASE_DIR, "Diagnostics")
ATTRIBUTES_PATH  = _resolver_ruta(BASE_DIR, "AttributesDictionary")

ECG_DIR_DENOISED = os.path.join(BASE_DIR, "ECGDataDenoised")
ECG_DIR_RAW      = os.path.join(BASE_DIR, "ECGData")
ECG_DIR          = ECG_DIR_DENOISED if os.path.isdir(ECG_DIR_DENOISED) else ECG_DIR_RAW

# Parámetros del dataset (Zheng 2020)
FS          = 500    # Hz — frecuencia nativa, sin resampleo
SIGNAL_LEN  = 5000   # muestras (10 s × 500 Hz)
N_LEADS     = 12

# Paralelismo para la carga/preprocesamiento masivo (ver nota de cabecera).
# 4 de 6 núcleos — mismo criterio que num_workers=4 en PTBXL_work/STAGE_3.
N_WORKERS   = 4

# Agrupación jerárquica 11 ritmos → 4 clases (Zheng 2020, Tabla 5)
RHYTHM_TO_CLASS = {
    "AFIB":  "AFIB",   # Fibrilación auricular
    "AF":    "AFIB",   # Flutter auricular
    "AT":    "GSVT",   # Taquicardia auricular
    "AVNRT": "GSVT",   # Reentrada nodal AV
    "AVRT":  "GSVT",   # Reentrada AV
    "SAAWR": "GSVT",   # Sinus atrium to atrial wandering rhythm
    "ST":    "GSVT",   # Taquicardia sinusal
    "SVT":   "GSVT",   # Taquicardia supraventricular
    "SB":    "SB",     # Bradicardia sinusal
    "SR":    "SR",     # Ritmo sinusal normal
    "SA":    "SR",     # Irregularidad sinusal (acrónimo real en Diagnostics: "SA", no "SI")
}

CLASS_NAMES  = ["AFIB", "GSVT", "SB", "SR"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
LEAD_NAMES   = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# Colores por clase (usados en todas las figuras)
CLASS_COLORS = {
    "AFIB": "#E63946",
    "GSVT": "#F4A261",
    "SB":   "#2A9D8F",
    "SR":   "#457B9D",
}


# =============================================================================
# 1. CARGA DE METADATOS
# =============================================================================

def load_diagnostics(path: str) -> pd.DataFrame:
    """
    Carga Diagnostics.csv/.xlsx y aplica la agrupación jerárquica
    11 ritmos → 4 clases (Zheng 2020).
    """
    df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print(f"[INFO] Diagnostics cargado: {len(df)} registros")
    print(f"       Columnas: {list(df.columns)}")

    # Ritmos originales presentes
    print(f"\n[INFO] Ritmos originales (11):")
    for ritmo, n in df["Rhythm"].value_counts().items():
        clase = RHYTHM_TO_CLASS.get(ritmo, "❌ sin mapeo")
        print(f"       {ritmo:8s} → {clase:6s}  ({n} registros)")

    # Mapeo → 4 clases
    df["Label"]    = df["Rhythm"].map(RHYTHM_TO_CLASS)
    df["LabelIdx"] = df["Label"].map(CLASS_TO_IDX)

    n_unmapped = df["Label"].isna().sum()
    if n_unmapped > 0:
        print(f"\n[WARN] {n_unmapped} registros sin mapeo → descartados")
        df = df.dropna(subset=["Label"]).reset_index(drop=True)

    return df


# =============================================================================
# 2. ETAPA 1 — EXPLORACIÓN
# =============================================================================

def explorar_dataset(df: pd.DataFrame, ecg_dir: str):
    """Estadísticas y figura de exploración inicial."""

    print("\n" + "="*60)
    print("  ETAPA 1 — EXPLORACIÓN CHAPMAN-SHAOXING")
    print("="*60)

    dist  = df["Label"].value_counts().reindex(CLASS_NAMES)
    total = len(df)

    # --- Distribución de clases ---
    print(f"\nTotal de registros usables: {total}")
    print("\nDistribución de clases agrupadas:")
    for cls, n in dist.items():
        bar = "█" * int(n / total * 40)
        print(f"  {cls:6s} │ {bar:<40s} {n:5d} ({n/total*100:.1f}%)")

    # --- Desequilibrio ---
    n_max, n_min = dist.max(), dist.min()
    print(f"\n  Ratio max/min de clases: {n_max/n_min:.2f}x  "
          f"({'moderado — weighted loss recomendable' if n_max/n_min < 2.5 else 'severo'})")

    # --- Demografía ---
    age_col = "PatientAge" if "PatientAge" in df.columns else None
    sex_col = "Gender"     if "Gender"     in df.columns else None
    if age_col:
        print(f"\n  Edad: media={df[age_col].mean():.1f}  "
              f"min={df[age_col].min()}  max={df[age_col].max()}")
    if sex_col:
        print(f"  Sexo: {df[sex_col].value_counts().to_dict()}")

    # --- Archivos CSV ---
    csv_files = [f for f in os.listdir(ecg_dir) if f.endswith(".csv")]
    print(f"\n  Archivos ECG en {os.path.basename(ecg_dir)}/: {len(csv_files)}")

    # Los CSV de Chapman-Shaoxing no traen fila de encabezado (header=None);
    # las 12 columnas siguen el orden fijo de LEAD_NAMES (Zheng 2020).
    sample_sig = pd.read_csv(os.path.join(ecg_dir, csv_files[0]), header=None)
    print(f"  Dimensiones de una señal: {sample_sig.shape}  "
          f"(esperado: {SIGNAL_LEN} × {N_LEADS})")
    print(f"  Derivaciones (orden fijo, sin encabezado en el CSV): {LEAD_NAMES}")

    # --- Figura ---
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

    # Panel superior: distribución de clases
    ax_dist = fig.add_subplot(gs[0, :])
    colors  = [CLASS_COLORS[c] for c in CLASS_NAMES]
    bars    = ax_dist.bar(CLASS_NAMES, dist.values, color=colors,
                          edgecolor="white", linewidth=0.8)
    ax_dist.set_title(
        "Distribución de clases — Chapman-Shaoxing (agrupación jerárquica, Zheng 2020)",
        fontsize=13, fontweight="bold")
    ax_dist.set_ylabel("Número de registros")
    for bar, val in zip(bars, dist.values):
        ax_dist.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 30,
                     f"{val}\n({val/total*100:.1f}%)",
                     ha="center", fontsize=10)
    ax_dist.spines[["top","right"]].set_visible(False)

    # Paneles inferiores: Lead II por clase
    t = np.arange(SIGNAL_LEN) / FS
    for i, cls in enumerate(CLASS_NAMES):
        ax  = fig.add_subplot(gs[1, i])
        row = df[df["Label"] == cls].iloc[0]
        fpath = os.path.join(ecg_dir, str(row["FileName"]) + ".csv")
        if os.path.exists(fpath):
            sig = pd.read_csv(fpath, header=None).values
            ax.plot(t, sig[:, 1], color=CLASS_COLORS[cls], linewidth=0.7)
            ax.set_title(cls, fontsize=11, fontweight="bold",
                         color=CLASS_COLORS[cls])
            ax.set_xlabel("Tiempo (s)", fontsize=9)
            ax.set_ylabel("Amplitud", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Archivo no encontrado",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cls)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle(
        f"Chapman-Shaoxing · Exploración inicial · Lead II · 500 Hz",
        fontsize=13, y=1.01)
    out = "bloque2_etapa1_exploracion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n[OK] Figura guardada: {out}")

    return dist


# =============================================================================
# 3. ETAPA 2 — PREPROCESAMIENTO
# =============================================================================
# Pipeline definitivo (decisiones documentadas en cabecera del script):
#   1. Filtro Butterworth paso-banda 0.5–40 Hz (bidireccional, sin desfase)
#   2. Z-score POR DERIVACIÓN + clip ±4σ
#
# Por qué por derivación y no global (bug encontrado en Etapa 3):
#   Un Z-score calculado sobre las 60.000 muestras del registro completo
#   (5000 × 12) queda dominado por la línea base, que es la inmensa
#   mayoría de los puntos. La std resultante es pequeña, así que los
#   picos QRS —pocos pero muy altos— se normalizan a 10-20σ en vez de
#   ~1σ. Esos valores extremos entran a la primera Conv1D, se acumulan
#   en el kernel, y explotan a inf antes de que el BatchNorm (que arranca
#   con running stats en 0/1) alcance a compensar → loss=NaN desde la
#   primera época → gradientes NaN → el modelo se queda prediciendo
#   siempre la clase mayoritaria.
#   Normalizar cada una de las 12 derivaciones con su propia media/std
#   refleja la variabilidad real de esa derivación (no la de las 60.000
#   muestras diluidas), y el clip ±4σ elimina outliers residuales sin
#   tocar la morfología clínica (los QRS reales rara vez superan 4σ
#   per-lead). Mismo esquema que usa PatchECG (normalización per-lead).
#
# Diferencia respecto al Bloque 1 (PTB-XL):
#   PTB-XL  → min-max  (tarea morfológica, amplitudes relativas importan)
#   Chapman → Z-score por derivación + clip ±4σ (tarea rítmica)
# =============================================================================

ZSCORE_CLIP_SIGMA = 4.0   # ver justificación arriba (PatchECG, per-lead)


def butter_bandpass(lowcut=0.5, highcut=40.0, fs=FS, order=4):
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return b, a


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Preprocesa una señal (5000, 12):
      1. Butterworth 0.5–40 Hz  bidireccional por derivación
      2. Z-score POR DERIVACIÓN (media/std propias de cada una de las 12
         columnas, no globales) + clip a ±4σ

    Returns: array float32 (5000, 12) normalizado
    """
    b, a = butter_bandpass()
    sig_filt = np.zeros_like(signal, dtype=np.float64)

    for lead in range(signal.shape[1]):
        sig_filt[:, lead] = filtfilt(b, a, signal[:, lead].astype(np.float64))

    # Z-score por derivación: cada columna con su propia media/std
    mu  = sig_filt.mean(axis=0, keepdims=True)   # (1, 12)
    std = sig_filt.std(axis=0, keepdims=True)    # (1, 12)
    std_safe = np.where(std > 1e-8, std, 1.0)
    sig_norm = np.where(std > 1e-8, (sig_filt - mu) / std_safe, sig_filt)

    # Clip ±4σ: elimina outliers extremos sin distorsionar la morfología
    sig_norm = np.clip(sig_norm, -ZSCORE_CLIP_SIGMA, ZSCORE_CLIP_SIGMA)

    return sig_norm.astype(np.float32)


def verificar_calidad(signal: np.ndarray) -> bool:
    """
    Descarta registros corruptos:
      · NaN o Inf en cualquier derivación (bug encontrado: algunos CSV
        crudos de Chapman-Shaoxing traen NaN en V1-V6 completos; con
        NaN presente, "(max - min) < 1e-6" nunca es True porque las
        comparaciones con NaN son siempre False, así que sin este check
        esos registros pasaban el control de calidad sin ser detectados
        y contaminaban el entrenamiento con NaN)
      · Alguna derivación completamente en cero
      · Amplitud total prácticamente nula (señal plana)
    """
    if np.isnan(signal).any() or np.isinf(signal).any():
        return False
    if np.any(np.all(signal == 0, axis=0)):
        return False
    if (signal.max() - signal.min()) < 1e-6:
        return False
    return True


def visualizar_preprocesamiento(df: pd.DataFrame, ecg_dir: str):
    """
    Muestra Lead II de una señal antes y después del preprocesamiento,
    para verificar visualmente que el filtro y la normalización son correctos.
    """
    row   = df.iloc[0]
    fpath = os.path.join(ecg_dir, str(row["FileName"]) + ".csv")
    if not os.path.exists(fpath):
        print(f"[WARN] No se encontró {fpath}")
        return

    sig_raw  = pd.read_csv(fpath, header=None).values.astype(np.float32)
    sig_proc = preprocess_signal(sig_raw)
    t        = np.arange(SIGNAL_LEN) / FS

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t, sig_raw[:, 1], color="#6C757D", linewidth=0.7)
    axes[0].set_title(
        f"Lead II — Señal original  |  {row['FileName']}  |  Clase: {row['Label']}",
        fontsize=11)
    axes[0].set_ylabel("Amplitud (unidades raw)")
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(t, sig_proc[:, 1],
                 color=CLASS_COLORS.get(row["Label"], "#333"), linewidth=0.7)
    axes[1].set_title(
        "Lead II — Señal preprocesada  "
        "(Butterworth 0.5–40 Hz + Z-score por derivación, clip ±4σ)",
        fontsize=11)
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Amplitud (Z-score)")
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = "bloque2_etapa2_preprocesamiento.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[OK] Figura guardada: {out}")


# =============================================================================
# 4. CARGA Y PREPROCESAMIENTO COMPLETO (paralelizado)
# =============================================================================

def _procesar_registro(args: tuple) -> dict:
    """
    Procesa un único registro: lee el CSV, valida dimensiones/calidad y
    aplica el pipeline de preprocesamiento.

    Vive a nivel de módulo (no anidada) porque ProcessPoolExecutor en
    Windows usa multiprocessing con método "spawn": cada proceso worker
    reimporta este archivo como módulo y necesita poder resolver la
    función por su nombre para deserializar las tareas — igual que ocurre
    con PTBXLDataset en PTBXL_work/STAGE_2.
    """
    filename, label_idx, ecg_dir = args
    fpath = os.path.join(ecg_dir, filename + ".csv")

    if not os.path.exists(fpath):
        return {"ok": False, "motivo": "no encontrado", "filename": filename}

    try:
        sig = pd.read_csv(fpath, header=None).values.astype(np.float32)
    except Exception as e:
        return {"ok": False, "motivo": f"error de lectura: {e}", "filename": filename}

    if sig.shape != (SIGNAL_LEN, N_LEADS):
        return {"ok": False, "motivo": f"dimensiones {sig.shape}", "filename": filename}

    if not verificar_calidad(sig):
        return {"ok": False, "motivo": "calidad (plana o con derivación en cero)", "filename": filename}

    return {"ok": True, "X": preprocess_signal(sig), "y": label_idx, "filename": filename}


def cargar_y_preprocesar(df: pd.DataFrame,
                          ecg_dir: str,
                          max_registros: int = None,
                          n_workers: int = N_WORKERS) -> tuple:
    """
    Carga todas las señales, aplica el pipeline y devuelve arrays listos
    para entrenar.

    La carga (E/S de ~10.500 CSV) + el filtrado Butterworth bidireccional
    por derivación se distribuyen entre varios procesos con
    ProcessPoolExecutor, siguiendo el mismo criterio de prestaciones que
    PTBXL_work/STAGE_3 (num_workers=4 de 6 núcleos disponibles).

    Args:
        max_registros : int o None — usar un número pequeño (ej. 200)
                        para prueba rápida antes de procesar todo el dataset.
        n_workers     : número de procesos en paralelo.

    Returns:
        X   : (N, 5000, 12)  float32
        y   : (N,)           int64
        ids : list[str]      IDs de pacientes incluidos
    """
    if max_registros:
        df = df.head(max_registros)

    tareas = [(str(row["FileName"]), int(row["LabelIdx"]), ecg_dir)
              for _, row in df.iterrows()]

    print(f"\n[INFO] Procesando {len(tareas)} señales con {n_workers} procesos "
          f"en paralelo (ProcessPoolExecutor)...")

    X, y, ids = [], [], []
    descartados = 0
    procesados = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for resultado in executor.map(_procesar_registro, tareas, chunksize=20):
            procesados += 1

            if resultado["ok"]:
                X.append(resultado["X"])
                y.append(resultado["y"])
                ids.append(resultado["filename"])
            else:
                descartados += 1
                if resultado["motivo"] not in ("no encontrado", "calidad (plana o con derivación en cero)"):
                    print(f"[WARN] {resultado['filename']}: {resultado['motivo']}")

            if procesados % 1000 == 0:
                print(f"  Procesados: {procesados}/{len(tareas)} | "
                      f"Válidos: {len(X)} | Descartados: {descartados}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"\n[OK] Preprocesamiento completo:")
    print(f"     Señales válidas : {len(X)}")
    print(f"     Descartadas     : {descartados}")
    print(f"     Shape X         : {X.shape}  → (N, T=5000, leads=12)")
    print(f"     Shape y         : {y.shape}")
    print(f"     Distribución final: "
          + " | ".join(f"{CLASS_NAMES[c]}:{n}"
                       for c, n in sorted(Counter(y).items())))
    return X, y, ids


# =============================================================================
# 5. DIVISIÓN TRAIN / VAL / TEST
# =============================================================================

def dividir_dataset(X, y, ids,
                    train=0.70, val=0.10, test=0.20,
                    seed=42) -> dict:
    """
    División estratificada 70/10/20.

    Chapman-Shaoxing tiene 1 registro por paciente → no hay riesgo
    de data leakage entre splits (a diferencia de PTB-XL con strat_fold).
    """
    # Split 1: (train+val) vs test
    X_tv, X_te, y_tv, y_te, ids_tv, ids_te = train_test_split(
        X, y, ids, test_size=test, stratify=y, random_state=seed)

    # Split 2: train vs val
    val_adj = val / (train + val)
    X_tr, X_va, y_tr, y_va, ids_tr, ids_va = train_test_split(
        X_tv, y_tv, ids_tv, test_size=val_adj, stratify=y_tv, random_state=seed)

    splits = {
        "train": {"X": X_tr, "y": y_tr, "ids": ids_tr},
        "val":   {"X": X_va, "y": y_va, "ids": ids_va},
        "test":  {"X": X_te, "y": y_te, "ids": ids_te},
    }

    print("\n[OK] División estratificada 70/10/20:")
    for name, data in splits.items():
        n    = len(data["y"])
        dist = " | ".join(f"{CLASS_NAMES[c]}:{cnt}"
                          for c, cnt in sorted(Counter(data["y"]).items()))
        print(f"  {name:6s}: {n:5d} muestras  [{dist}]")

    return splits


# =============================================================================
# 6. CALCULAR PESOS DE CLASE (para CrossEntropyLoss)
# =============================================================================

def calcular_pesos_clase(y_train: np.ndarray) -> np.ndarray:
    """
    Pesos inversamente proporcionales a la frecuencia de cada clase.
    Se pasan a torch.nn.CrossEntropyLoss(weight=...) durante el entrenamiento.

    Fórmula: w_c = N_total / (N_clases × N_c)
    """
    counts = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = len(y_train) / (len(CLASS_NAMES) * counts)
    weights = weights / weights.sum() * len(CLASS_NAMES)   # renormalizar

    print("\n[OK] Pesos de clase para CrossEntropyLoss:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:6s}: {weights[i]:.4f}  (N={int(counts[i])})")

    return weights


# =============================================================================
# 7. GUARDAR ARRAYS PREPROCESADOS
# =============================================================================

def guardar_splits(splits: dict, output_dir: str):
    """Guarda X, y e IDs de cada split como .npy / .csv."""
    os.makedirs(output_dir, exist_ok=True)

    for name, data in splits.items():
        np.save(os.path.join(output_dir, f"shaoxing_{name}_X.npy"), data["X"])
        np.save(os.path.join(output_dir, f"shaoxing_{name}_y.npy"), data["y"])
        pd.Series(data["ids"]).to_csv(
            os.path.join(output_dir, f"shaoxing_{name}_ids.csv"), index=False)

    # Guardar pesos de clase
    weights = calcular_pesos_clase(splits["train"]["y"])
    np.save(os.path.join(output_dir, "shaoxing_class_weights.npy"), weights)

    print(f"\n[OK] Arrays guardados en: {output_dir}")
    print("     shaoxing_train_X.npy  · shaoxing_train_y.npy  · shaoxing_train_ids.csv")
    print("     shaoxing_val_X.npy    · shaoxing_val_y.npy    · shaoxing_val_ids.csv")
    print("     shaoxing_test_X.npy   · shaoxing_test_y.npy   · shaoxing_test_ids.csv")
    print("     shaoxing_class_weights.npy")


# =============================================================================
# 8. MAIN — PIPELINE COMPLETO
# =============================================================================

if __name__ == "__main__":

    print("="*60)
    print("  BLOQUE 2 — CHAPMAN-SHAOXING · Etapas 1 y 2")
    print("="*60)
    print(f"  ECG dir  : {ECG_DIR}")
    print(f"  Fs       : {FS} Hz (nativo, sin resampleo)")
    print(f"  Señal    : {SIGNAL_LEN} muestras × {N_LEADS} derivaciones")
    print(f"  Norma    : Z-score por derivación + clip ±{ZSCORE_CLIP_SIGMA}σ")
    print(f"  Pérdida  : CrossEntropyLoss con pesos de clase")

    # ------------------------------------------------------------------
    # 8.1  Metadatos
    # ------------------------------------------------------------------
    df = load_diagnostics(DIAGNOSTICS_PATH)

    # ------------------------------------------------------------------
    # 8.2  Etapa 1: Exploración
    # ------------------------------------------------------------------
    dist = explorar_dataset(df, ECG_DIR)

    # ------------------------------------------------------------------
    # 8.3  Verificación visual del preprocesamiento
    # ------------------------------------------------------------------
    visualizar_preprocesamiento(df, ECG_DIR)

    # ------------------------------------------------------------------
    # 8.4  Etapa 2: Preprocesamiento completo
    #
    #  ⚠ Para una prueba rápida, cambia max_registros=200
    #    El procesamiento completo (~10.500 archivos) se distribuye entre
    #    N_WORKERS procesos (ver cabecera) — más rápido que un for secuencial,
    #    pero sigue tardando varios minutos por el filtrado Butterworth.
    # ------------------------------------------------------------------
    X, y, ids = cargar_y_preprocesar(
        df, ECG_DIR,
        max_registros=None   # ← cambiar a 200 para prueba rápida
    )

    # ------------------------------------------------------------------
    # 8.5  División 70/10/20
    # ------------------------------------------------------------------
    splits = dividir_dataset(X, y, ids)

    # ------------------------------------------------------------------
    # 8.6  Guardar
    # ------------------------------------------------------------------
    OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed")
    guardar_splits(splits, output_dir=OUTPUT_DIR)

    print("\n✓ Etapas 1 y 2 del Bloque 2 completadas.")
    print("  Siguiente → Etapa 3: CNN1D / CNN-LSTM / Transformer sobre Chapman 500 Hz")
    print(f"  Clases    : {CLASS_NAMES}")
    print(f"  Función de pérdida: CrossEntropyLoss (multiclase, softmax)")
    print(f"  Diferencia vs Bloque 1: BCEWithLogitsLoss → CrossEntropyLoss")