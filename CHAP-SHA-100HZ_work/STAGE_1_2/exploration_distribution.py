# =============================================================================
# BLOQUE 3 — CHAPMAN-SHAOXING 100 Hz
# Etapa 1: Exploración del dataset
# Etapa 2: Preprocesamiento (pipeline definitivo)
#
# Réplica del Bloque 2 (Chapman-Shaoxing 500 Hz nativo) con un único cambio
# estructural: la señal se resamplea a 100 Hz para poder compararse en
# igualdad de condiciones con PTB-XL (Bloque 1, también 100 Hz) en el
# experimento cruzado del Bloque 4.
#
# Decisiones metodológicas documentadas:
#   · Resampleo 500 Hz → 100 Hz con filtro antialiasing polifásico
#     (scipy.signal.resample_poly, up=1/down=5) — ver "RESAMPLEO" abajo
#   · Filtro Butterworth paso-banda 0.5–40 Hz bidireccional, aplicado
#     DESPUÉS del resampleo (sobre la señal ya a 100 Hz)
#   · Recorte del primer 1 s POST-FILTRO, igual que el Bloque 2 — ver
#     "TRANSITORIO DE BORDE" abajo
#   · Normalización Z-score por derivación + clip ±4σ (idéntico al
#     Bloque 2 corregido — PatchECG; ver Etapa 2)
#   · División estratificada 70/10/20 con semilla fija (reproducible)
#   · CrossEntropyLoss + softmax — tarea multiclase (1 ritmo por paciente)
#
# RESAMPLEO 500 Hz → 100 Hz (FS_TARGET):
#   El dataset Chapman-Shaoxing se distribuye nativamente a 500 Hz (Zheng
#   2020), pero PTB-XL (Bloque 1) usa 100 Hz. Para que la comparación
#   cruzada del Bloque 4 (mismo backbone entrenado en dos datasets) no esté
#   confundida por la frecuencia de muestreo, este bloque decima
#   Chapman-Shaoxing a 100 Hz ANTES de filtrar.
#     · Método: scipy.signal.resample_poly(signal, up=1, down=5, axis=0)
#       — filtro FIR antialiasing polifásico aplicado por derivación
#       (axis=0 procesa cada una de las 12 columnas de forma independiente,
#       equivalente a un bucle por derivación pero vectorizado).
#     · Factor de decimación: 500/100 = 5 → up=1, down=5.
#     · 5000 muestras (10 s × 500 Hz) → 1000 muestras (10 s × 100 Hz).
#     · Justificado en Guo et al. (s41598-026-54206-8): "down sampled from
#       500 Hz to 100 Hz using a polyphase anti-aliasing filter".
#     · Respaldo adicional: PatchECG (s41598-026-41549-5) resamplea
#       Chapman-Shaoxing a 100 Hz para unificar datasets en un mismo
#       backbone — mismo objetivo que este Bloque 3 respecto al Bloque 4.
#
# TRANSITORIO DE BORDE DEL FILTRO (idéntico al Bloque 2, escalado a 100 Hz):
#   El Bloque 2 (500 Hz) descubrió que filtfilt (zero-phase) introduce un
#   transitorio de borde de ~1.0–1.4 s cuando el corte bajo del pasabanda
#   es 0.5 Hz — ver cabecera de
#   CHAP-SHA-500HZ_work/STAGE_1_2/exploration_distribution.py. Esa duración
#   depende del corte en Hz del filtro (constante de tiempo ~1/0.5 Hz), NO
#   de la frecuencia de muestreo: el mismo transitorio de ~1 s aparece aquí
#   aunque el filtro se aplique sobre la señal ya decimada a 100 Hz — solo
#   que representado en menos muestras (100 en vez de 500).
#   Por fidelidad al Bloque 2, este Bloque 3 aplica el mismo recorte,
#   escalado a la nueva fs: se descarta 1 s (100 muestras a 100 Hz) del
#   INICIO de la señal ya filtrada. Ventana final: 9 s = 900 muestras
#   —misma duración en segundos que el Bloque 2 (4500 muestras a 500 Hz =
#   9 s)—, aunque ya no coincide en número de muestras con PTB-XL (1000)
#   para el Bloque 4. TRIM_START_SAMPLES se resta ANTES del z-score, para
#   que la media/std por derivación no queden sesgadas por el transitorio.
#
# Ajuste de prestaciones (mismo criterio que PTBXL_work y Bloque 2):
#   · Máquina: Ryzen 5 5600X (6 núcleos / 12 hilos) + RTX 5050 8 GB VRAM
#   · El resampleo + filtrado Butterworth bidireccional (× 12 derivaciones)
#     de ~10.500 señales se distribuye con ProcessPoolExecutor(max_workers=4)
#     — mismo criterio que num_workers=4 en PTBXL_work/STAGE_3 y en el
#     Bloque 2.
#
# Referencias:
#   Zheng et al. (2020) Scientific Data 7:48
#   Guo et al., Scientific Reports (s41598-026-54206-8) — resampleo 500→100 Hz
#   PatchECG (s41598-026-41549-5) — unificación de datasets a 100 Hz
# =============================================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # cada figura se genera 2 veces (es/en) en lote, sin ventana interactiva
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.model_selection import train_test_split
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# =============================================================================

# Dataset crudo: mismo origen que el Bloque 2 (500 Hz nativo). El
# resampleo a 100 Hz ocurre en el pipeline, no en el origen de los datos.
BASE_DIR = r"C:\Users\Lina Castañeda\Documents\Originals\CHAPMAN-SHAOXING"

# Carpeta donde se guardan los arrays preprocesados (.npy), FUERA del
# repositorio del proyecto — igual criterio que el Bloque 2, que guarda
# su "preprocessed/" junto al dataset crudo en vez de dentro de
# CHAP-SHA-500HZ_work/. Aquí se usa un sufijo "_100hz" para no chocar con
# la carpeta "preprocessed/" del Bloque 2 (500 Hz), que vive en el mismo
# BASE_DIR (dataset crudo compartido entre ambos bloques).
PREPROCESSED_OUTPUT_DIR = Path(BASE_DIR) / "preprocessed_100hz"


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

# Parámetros del dataset crudo (Zheng 2020) — frecuencia nativa de los CSV
FS_NATIVE      = 500    # Hz — frecuencia de muestreo nativa de los archivos
SIGNAL_LEN_RAW = 5000   # muestras crudas por registro (10 s × 500 Hz)
N_LEADS        = 12

# Parámetros de destino tras el resampleo (ver "RESAMPLEO" en cabecera).
# FS_TARGET=100: fijado para compatibilidad con PTB-XL (Bloque 1) y el
# experimento cruzado del Bloque 4; respaldado por PatchECG
# (s41598-026-41549-5), que resamplea Chapman a 100 Hz con el mismo fin.
FS_TARGET  = 100    # Hz — frecuencia de muestreo destino (igual que PTB-XL)
SIGNAL_LEN = 1000   # muestras tras resampleo (10 s × 100 Hz)

# Factor de decimación polifásica: 500/100 = 5 → up=1, down=5
# (scipy.signal.resample_poly; ver Guo et al., s41598-026-54206-8)
RESAMPLE_UP   = 1
RESAMPLE_DOWN = 5

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

# Colores por clase (usados en todas las figuras). Paleta fija del
# proyecto (no generica), validada con el validador de paletas
# categoricas (banda de luminosidad, piso de croma, separacion bajo
# daltonismo protan/deutan, piso de vision normal) — mismos 4 primeros
# slots que PTBXL_work y el Bloque 2, para consistencia visual entre bloques.
CLASS_COLORS = {
    "AFIB": "darkorchid",
    "GSVT": "seagreen",
    "SB":   "royalblue",
    "SR":   "darkorange",
}

OUTPUT_DIR    = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR_ES = OUTPUT_DIR / "es"
OUTPUT_DIR_EN = OUTPUT_DIR / "en"


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

TEXTS = {
    "es": {
        "dist_title": "Distribución de clases — Chapman-Shaoxing (agrupación jerárquica, Zheng 2020)",
        "dist_ylabel": "Número de registros",
        "lead_xlabel": "Tiempo (s)",
        "lead_ylabel": "Amplitud",
        "not_found": "Archivo no encontrado",
        "suptitle": "Chapman-Shaoxing · Exploración inicial · Lead II · señal cruda 500 Hz (se resamplea a 100 Hz en la Etapa 2)",
        "raw_title": "Lead II — Señal original 500 Hz  |  {fname}  |  Clase: {label}",
        "raw_ylabel": "Amplitud (unidades raw)",
        "proc_title": "Lead II — Señal preprocesada a 100 Hz (9 s)  (resampleo polifásico + Butterworth 0.5–40 Hz + recorte de 1 s + Z-score por derivación, clip ±4σ)",
        "proc_ylabel": "Amplitud (Z-score)",
        "trim_label": "1 s descartado\n(transitorio de borde del filtro)",
        "xlabel": "Tiempo (s)",
    },
    "en": {
        "dist_title": "Class distribution — Chapman-Shaoxing (hierarchical grouping, Zheng 2020)",
        "dist_ylabel": "Number of records",
        "lead_xlabel": "Time (s)",
        "lead_ylabel": "Amplitude",
        "not_found": "File not found",
        "suptitle": "Chapman-Shaoxing · Initial exploration · Lead II · raw 500 Hz signal (resampled to 100 Hz in Stage 2)",
        "raw_title": "Lead II — Original 500 Hz signal  |  {fname}  |  Class: {label}",
        "raw_ylabel": "Amplitude (raw units)",
        "proc_title": "Lead II — Preprocessed signal at 100 Hz (9 s)  (polyphase resampling + Butterworth 0.5–40 Hz + 1 s trim + per-lead Z-score, clip ±4σ)",
        "proc_ylabel": "Amplitude (Z-score)",
        "trim_label": "1 s discarded\n(filter edge transient)",
        "xlabel": "Time (s)",
    },
}


def explorar_dataset(df: pd.DataFrame, ecg_dir: str, lang: str = "es", out_dir: Path = None):
    """Estadísticas y figura de exploración inicial (sobre la señal cruda a 500 Hz)."""
    t_ = TEXTS[lang]

    print("\n" + "="*60)
    print("  ETAPA 1 — EXPLORACIÓN CHAPMAN-SHAOXING (destino: 100 Hz)")
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
    print(f"  Dimensiones de una señal cruda: {sample_sig.shape}  "
          f"(esperado: {SIGNAL_LEN_RAW} × {N_LEADS} a {FS_NATIVE} Hz)")
    print(f"  Tras resampleo+recorte (Etapa 2): {TRIMMED_SIGNAL_LEN} × {N_LEADS} a {FS_TARGET} Hz")
    print(f"  Derivaciones (orden fijo, sin encabezado en el CSV): {LEAD_NAMES}")

    # --- Figura ---
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.35)

    # Panel superior: distribución de clases
    ax_dist = fig.add_subplot(gs[0, :])
    colors  = [CLASS_COLORS[c] for c in CLASS_NAMES]
    bars    = ax_dist.bar(CLASS_NAMES, dist.values, color=colors,
                          edgecolor="white", linewidth=0.8)
    ax_dist.set_title(t_["dist_title"], fontsize=13, fontweight="bold")
    ax_dist.set_ylabel(t_["dist_ylabel"])
    for bar, val in zip(bars, dist.values):
        ax_dist.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 30,
                     f"{val}\n({val/total*100:.1f}%)",
                     ha="center", fontsize=10)
    ax_dist.spines[["top","right"]].set_visible(False)

    # Paneles inferiores: Lead II por clase (señal cruda, 500 Hz)
    t = np.arange(SIGNAL_LEN_RAW) / FS_NATIVE
    for i, cls in enumerate(CLASS_NAMES):
        ax  = fig.add_subplot(gs[1, i])
        row = df[df["Label"] == cls].iloc[0]
        fpath = os.path.join(ecg_dir, str(row["FileName"]) + ".csv")
        if os.path.exists(fpath):
            sig = pd.read_csv(fpath, header=None).values
            ax.plot(t, sig[:, 1], color=CLASS_COLORS[cls], linewidth=0.7)
            ax.set_title(cls, fontsize=11, fontweight="bold",
                         color=CLASS_COLORS[cls])
            ax.set_xlabel(t_["lead_xlabel"], fontsize=9)
            ax.set_ylabel(t_["lead_ylabel"], fontsize=9)
        else:
            ax.text(0.5, 0.5, t_["not_found"],
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cls)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle(t_["suptitle"], fontsize=13, y=1.01)
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bloque3_etapa1_exploracion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Figura guardada: {out}")

    return dist


# =============================================================================
# 3. ETAPA 2 — PREPROCESAMIENTO
# =============================================================================
# Pipeline definitivo (decisiones documentadas en cabecera del script):
#   1. Resampleo polifásico 500 Hz → 100 Hz (resample_poly, up=1/down=5,
#      por derivación) — ver "RESAMPLEO" en cabecera
#   2. Filtro Butterworth paso-banda 0.5–40 Hz (bidireccional, sin desfase),
#      aplicado SOBRE LA SEÑAL YA RESAMPLEADA a 100 Hz (Nyquist=50 Hz, el
#      corte de 40 Hz sigue siendo válido)
#   3. Z-score POR DERIVACIÓN + clip ±4σ (idéntico al Bloque 2 corregido)
#
# Por qué por derivación y no global (mismo hallazgo que el Bloque 2):
#   Un Z-score calculado sobre las muestras del registro completo queda
#   dominado por la línea base, que es la inmensa mayoría de los puntos.
#   La std resultante es pequeña, así que los picos QRS —pocos pero muy
#   altos— se normalizan a 10-20σ en vez de ~1σ. Esos valores extremos
#   entran a la primera Conv1D, se acumulan en el kernel, y explotan a inf
#   antes de que el BatchNorm alcance a compensar → loss=NaN desde la
#   primera época. Normalizar cada una de las 12 derivaciones con su propia
#   media/std, más el clip ±4σ, evita este problema sin distorsionar la
#   morfología clínica. Mismo esquema que PatchECG (normalización per-lead).
#
# Diferencia respecto al Bloque 1 (PTB-XL) y al Bloque 2 (Chapman 500 Hz):
#   PTB-XL   → min-max            (tarea morfológica)
#   Chapman  → Z-score por derivación + clip ±4σ (tarea rítmica) — igual
#              en el Bloque 2 (500 Hz) y este Bloque 3 (100 Hz)
# =============================================================================

ZSCORE_CLIP_SIGMA = 4.0   # ver justificación arriba (PatchECG, per-lead)

# Recorte del transitorio de borde del filtro (ver "TRANSITORIO DE BORDE"
# en la cabecera del script): se descarta 1 s (100 muestras a 100 Hz) del
# INICIO de la señal ya resampleada y filtrada. Ventana final: 9 s = 900
# muestras. TRIM_START_SAMPLES se resta ANTES del z-score para que la
# media/std por derivación no queden sesgadas por el transitorio.
TRIM_START_SAMPLES = 100                          # 1 s a 100 Hz
TRIMMED_SIGNAL_LEN = SIGNAL_LEN - TRIM_START_SAMPLES  # 900 muestras = 9 s


def resample_to_target(signal: np.ndarray) -> np.ndarray:
    """
    Resamplea una señal cruda (5000, 12) a 100 Hz mediante un filtro
    antialiasing polifásico (scipy.signal.resample_poly).

    axis=0 aplica el resampleo de forma independiente a cada una de las
    12 columnas (derivaciones) — equivalente a un bucle por derivación,
    pero vectorizado.

    Returns: array (1000, 12) — 10 s a 100 Hz
    """
    return resample_poly(signal, up=RESAMPLE_UP, down=RESAMPLE_DOWN, axis=0)


def butter_bandpass(lowcut=0.5, highcut=40.0, fs=FS_TARGET, order=4):
    nyq  = fs / 2.0
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return b, a


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """
    Preprocesa una señal cruda (5000, 12) a 500 Hz:
      1. Resampleo polifásico a 100 Hz → (1000, 12)  (ver "RESAMPLEO")
      2. Butterworth 0.5–40 Hz bidireccional por derivación, sobre la
         señal ya resampleada (Nyquist=50 Hz a 100 Hz)
      3. Recorte del primer 1 s (100 muestras) — transitorio de borde
         del filtro (ver "TRANSITORIO DE BORDE" en la cabecera del script)
      4. Z-score POR DERIVACIÓN (media/std propias de cada una de las 12
         columnas, no globales, calculadas YA SOBRE LA SEÑAL RECORTADA)
         + clip a ±4σ

    Returns: array float32 (900, 12) normalizado — ventana de 9 s a 100 Hz
    """
    sig_100hz = resample_to_target(signal)          # (1000, 12)

    b, a = butter_bandpass()
    sig_filt = np.zeros_like(sig_100hz, dtype=np.float64)
    for lead in range(sig_100hz.shape[1]):
        sig_filt[:, lead] = filtfilt(b, a, sig_100hz[:, lead].astype(np.float64))

    # Recorte del transitorio de borde ANTES de calcular media/std, para
    # que el z-score refleje la señal ya estable (ver TRIM_START_SAMPLES)
    sig_filt = sig_filt[TRIM_START_SAMPLES:]         # (900, 12)

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
    Descarta registros corruptos (chequeo sobre la señal CRUDA, antes del
    resampleo — mismo criterio que el Bloque 2):
      · NaN o Inf en cualquier derivación (algunos CSV crudos de
        Chapman-Shaoxing traen NaN en V1-V6 completos; sin este check
        esos registros contaminarían el resampleo/filtrado con NaN)
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


def visualizar_preprocesamiento(df: pd.DataFrame, ecg_dir: str, lang: str = "es",
                                out_dir: Path = None):
    """
    Muestra Lead II de una señal antes (500 Hz cruda) y después (100 Hz
    preprocesada, recortada a 9 s) del pipeline, para verificar
    visualmente que el resampleo, el filtro, el recorte del transitorio
    de borde y la normalización son correctos.

    El panel superior (señal cruda, 500 Hz) usa el eje de tiempo original
    0-10s y sombrea el 1er segundo descartado. El panel inferior (señal
    preprocesada, 9s = 900 muestras a 100 Hz) se grafica en el eje 1-10s,
    para que ambos paneles queden alineados en el tiempo real del
    registro — ver "TRANSITORIO DE BORDE DEL FILTRO" en la cabecera del
    script.
    """
    t_ = TEXTS[lang]
    row   = df.iloc[0]
    fpath = os.path.join(ecg_dir, str(row["FileName"]) + ".csv")
    if not os.path.exists(fpath):
        print(f"[WARN] No se encontró {fpath}")
        return

    sig_raw  = pd.read_csv(fpath, header=None).values.astype(np.float32)
    sig_proc = preprocess_signal(sig_raw)                       # (900, 12)
    trim_s   = TRIM_START_SAMPLES / FS_TARGET                   # 1.0 s
    t_raw    = np.arange(SIGNAL_LEN_RAW) / FS_NATIVE             # 0..10s @ 500 Hz
    t_proc   = trim_s + np.arange(TRIMMED_SIGNAL_LEN) / FS_TARGET  # 1..10s @ 100 Hz

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)

    axes[0].plot(t_raw, sig_raw[:, 1], color="#6C757D", linewidth=0.7)
    axes[0].axvspan(0, trim_s, color="#B22222", alpha=0.12, zorder=0)
    axes[0].axvline(trim_s, color="#B22222", linewidth=1.0,
                    linestyle="--", alpha=0.6)
    axes[0].text(trim_s / 2, axes[0].get_ylim()[1] * 0.9,
                t_["trim_label"], ha="center", va="top", fontsize=8, color="#B22222")
    axes[0].set_title(
        t_["raw_title"].format(fname=row["FileName"], label=row["Label"]),
        fontsize=11)
    axes[0].set_ylabel(t_["raw_ylabel"])
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(t_proc, sig_proc[:, 1],
                 color=CLASS_COLORS.get(row["Label"], "#333"), linewidth=0.7)
    axes[1].axvspan(0, trim_s, color="#B22222", alpha=0.06, zorder=0)
    axes[1].set_title(t_["proc_title"], fontsize=11)
    axes[1].set_xlabel(t_["xlabel"])
    axes[1].set_ylabel(t_["proc_ylabel"])
    axes[1].set_xlim(0, SIGNAL_LEN_RAW / FS_NATIVE)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out_dir = out_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "bloque3_etapa2_preprocesamiento.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figura guardada: {out}")


# =============================================================================
# 4. CARGA Y PREPROCESAMIENTO COMPLETO (paralelizado)
# =============================================================================

def _procesar_registro(args: tuple) -> dict:
    """
    Procesa un único registro: lee el CSV, valida dimensiones/calidad
    (sobre la señal cruda a 500 Hz) y aplica el pipeline de
    resampleo + filtrado + normalización.

    Vive a nivel de módulo (no anidada) porque ProcessPoolExecutor en
    Windows usa multiprocessing con método "spawn": cada proceso worker
    reimporta este archivo como módulo y necesita poder resolver la
    función por su nombre para deserializar las tareas.
    """
    filename, label_idx, ecg_dir = args
    fpath = os.path.join(ecg_dir, filename + ".csv")

    if not os.path.exists(fpath):
        return {"ok": False, "motivo": "no encontrado", "filename": filename}

    try:
        sig = pd.read_csv(fpath, header=None).values.astype(np.float32)
    except Exception as e:
        return {"ok": False, "motivo": f"error de lectura: {e}", "filename": filename}

    if sig.shape != (SIGNAL_LEN_RAW, N_LEADS):
        return {"ok": False, "motivo": f"dimensiones {sig.shape}", "filename": filename}

    if not verificar_calidad(sig):
        return {"ok": False, "motivo": "calidad (plana o con derivación en cero)", "filename": filename}

    return {"ok": True, "X": preprocess_signal(sig), "y": label_idx, "filename": filename}


def cargar_y_preprocesar(df: pd.DataFrame,
                          ecg_dir: str,
                          max_registros: int = None,
                          n_workers: int = N_WORKERS) -> tuple:
    """
    Carga todas las señales, aplica el pipeline (resampleo + filtro +
    normalización) y devuelve arrays listos para entrenar.

    La carga (E/S de ~10.500 CSV) + el resampleo + el filtrado Butterworth
    bidireccional por derivación se distribuyen entre varios procesos con
    ProcessPoolExecutor, siguiendo el mismo criterio de prestaciones que
    PTBXL_work/STAGE_3 y el Bloque 2 (num_workers=4 de 6 núcleos disponibles).

    Args:
        max_registros : int o None — usar un número pequeño (ej. 200)
                        para prueba rápida antes de procesar todo el dataset.
        n_workers     : número de procesos en paralelo.

    Returns:
        X   : (N, 900, 12)  float32  — 9 s a 100 Hz (tras resampleo + recorte)
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
    print(f"     Shape X         : {X.shape}  → (N, T={TRIMMED_SIGNAL_LEN}, leads={N_LEADS}) @ {FS_TARGET} Hz")
    print(f"     Shape y         : {y.shape}")
    print(f"     Distribución final: "
          + " | ".join(f"{CLASS_NAMES[c]}:{n}"
                       for c, n in sorted(Counter(y).items())))

    # Verificación del Bloque 3: tras resampleo + recorte del transitorio,
    # X debe tener TRIMMED_SIGNAL_LEN (900) muestras por derivación y el
    # clip ±4σ debe garantizar |X| <= 4.0.
    assert X.shape[1:] == (TRIMMED_SIGNAL_LEN, N_LEADS), (
        f"Shape inesperado tras resampleo+recorte: {X.shape[1:]}, se esperaba "
        f"({TRIMMED_SIGNAL_LEN}, {N_LEADS})")
    max_abs = float(np.abs(X).max())
    assert max_abs <= ZSCORE_CLIP_SIGMA, (
        f"Valor máximo absoluto {max_abs:.4f} excede el clip ±{ZSCORE_CLIP_SIGMA}σ")
    print(f"     Verificación    : X.shape={X.shape}  |X|max={max_abs:.4f} "
          f"(<= {ZSCORE_CLIP_SIGMA}) ✓")

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
    print("  BLOQUE 3 — CHAPMAN-SHAOXING 100 Hz · Etapas 1 y 2")
    print("="*60)
    print(f"  ECG dir       : {ECG_DIR}")
    print(f"  Fs nativo     : {FS_NATIVE} Hz  →  Fs destino: {FS_TARGET} Hz")
    print(f"  Resampleo     : polifásico (resample_poly, up={RESAMPLE_UP}/down={RESAMPLE_DOWN})")
    print(f"  Señal         : {SIGNAL_LEN_RAW} muestras crudas @ {FS_NATIVE} Hz -> "
          f"{SIGNAL_LEN} muestras @ {FS_TARGET} Hz -> recorte de "
          f"{TRIM_START_SAMPLES} (transitorio de borde) -> "
          f"{TRIMMED_SIGNAL_LEN} muestras × {N_LEADS} derivaciones (9 s)")
    print(f"  Norma         : Z-score por derivación + clip ±{ZSCORE_CLIP_SIGMA}σ")
    print(f"  Pérdida       : CrossEntropyLoss con pesos de clase")

    OUTPUT_DIR_ES.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR_EN.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 8.1  Metadatos
    # ------------------------------------------------------------------
    df = load_diagnostics(DIAGNOSTICS_PATH)

    # ------------------------------------------------------------------
    # 8.2  Etapa 1: Exploración (figuras ES/EN)
    # ------------------------------------------------------------------
    dist = explorar_dataset(df, ECG_DIR, lang="es", out_dir=OUTPUT_DIR_ES)
    explorar_dataset(df, ECG_DIR, lang="en", out_dir=OUTPUT_DIR_EN)

    # ------------------------------------------------------------------
    # 8.3  Verificación visual del preprocesamiento (figuras ES/EN)
    # ------------------------------------------------------------------
    visualizar_preprocesamiento(df, ECG_DIR, lang="es", out_dir=OUTPUT_DIR_ES)
    visualizar_preprocesamiento(df, ECG_DIR, lang="en", out_dir=OUTPUT_DIR_EN)

    # ------------------------------------------------------------------
    # 8.4  Etapa 2: Preprocesamiento completo
    #
    #  ⚠ Para una prueba rápida, cambia max_registros=200
    #    El procesamiento completo (~10.500 archivos) se distribuye entre
    #    N_WORKERS procesos (ver cabecera) — más rápido que un for secuencial,
    #    pero sigue tardando varios minutos por el resampleo + filtrado.
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
    # 8.6  Guardar (fuera del proyecto: Documents/Originals/CHAPMAN-SHAOXING/
    #      preprocessed_100hz/ — ver PREPROCESSED_OUTPUT_DIR en cabecera)
    # ------------------------------------------------------------------
    guardar_splits(splits, output_dir=str(PREPROCESSED_OUTPUT_DIR))

    print("\n✓ Etapas 1 y 2 del Bloque 3 completadas.")
    print(f"  Graficos + resumen ES en: {OUTPUT_DIR_ES}")
    print(f"  Graficos + resumen EN en: {OUTPUT_DIR_EN}")
    print(f"  Preprocesados en: {PREPROCESSED_OUTPUT_DIR}")
    print("  Siguiente → Etapa 3: CNN1D / CNN-LSTM / Transformer sobre Chapman 100 Hz")
    print(f"  Clases    : {CLASS_NAMES}")
    print(f"  Función de pérdida: CrossEntropyLoss (multiclase, softmax)")
