# Deeplearning_ECG_Exploration: rama `thesis`

Código y resultados del trabajo de grado *"Exploración de métodos de aprendizaje profundo para la detección de cardiopatías en señales de electrocardiograma"*, de Lina María Castañeda Cárdenas (Ingeniería Electrónica, Universidad de los Llanos). Director: Angel Alfonso Cruz Roa.

## Estructura

| Carpeta | Contenido | Capítulo del libro |
|---|---|---|
| `PTBXL_work/STAGE_1` | Análisis exploratorio y pesos de clase de PTB-XL | 7.1 |
| `PTBXL_work/STAGE_2` | Pipeline de preprocesamiento multietiqueta (opción `use_clinical` para edad y sexo) | 6 |
| `PTBXL_work/STAGE_3` | Comparación de arquitecturas: CNN1D-ResNet, CNN-LSTM V1/V2 y ECG Transformer V1/V2 | 7.2 |
| `PTBXL_work/STAGE_4` | Comparación de los cinco modelos de PTB-XL | 7.2 |
| `PTBXL_work/STAGE_5_ENCODER_ECG` | Modelos reorganizados en codificador (128-d) + cabeza y reentrenados **solo con ECG**. Es la línea base del experimento clínico | 7 (información clínica) |
| `PTBXL_work/STAGE_5_ENCODER_CLINICO` | Los mismos modelos con **edad y sexo** concatenados al embedding (130-d) | 7 (información clínica) |
| `CHAP-SHA-500HZ_work` | Chapman-Shaoxing a 500 Hz y comparación entre bloques | 7.3, 7.5 |
| `CHAP-SHA-100HZ_work` | Chapman-Shaoxing remuestreado a 100 Hz | 7.4 |
| `CROSS_VALIDATION` | Comparación 500 vs. 100 Hz y evaluación cruzada PTB-XL → Chapman-Shaoxing | 7.4, 7.6 |

**Importante:** `STAGE_3` y `STAGE_5_ENCODER_ECG` son entrenamientos distintos de las mismas arquitecturas, por eso sus cifras difieren un poco (por ejemplo, F1 macro de la CNN1D-ResNet: 0,713 frente a 0,704). `STAGE_3` responde qué arquitectura es mejor. `STAGE_5_*` responde si la edad y el sexo aportan, comparando dos modelos con exactamente la misma estructura.

## Rama `sipaim`

El experimento de transferencia de aprendizaje (sondeo lineal de los codificadores de PTB-XL en Chapman-Shaoxing) y el artículo aceptado en SIPAIM 2026 están en la rama `sipaim`, con la etiqueta `sipaim2026`. En el libro se presentan como resultado adicional (capítulo 10).
