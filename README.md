# Clasificación de granos de cacao (Proyecto final IA): InspectIA

Dataset: "Ecuadorian cacao beans for classification" (Kaggle).
1634 granos únicos: 1261 `sana` y 373 `mala` (enfermos). Recolectados en La Maná, Ecuador.

Licencia del dataset: CC BY-NC 4.0 (uso académico citando a los autores; no permite uso comercial).
Fuente: Zhinin-Vera, L. et al. (2023). Artificial Vision Technique to Detect and Classify Cocoa Beans. IWANN 2023, LNCS 14135, Springer. DOI 10.1007/978-3-031-43078-7_18.

## Descarga del dataset

El dataset **no se versiona** en el repositorio por su tamaño (el .zip pesa 5,15 GB); el profesor confirmó que basta el enlace. Cada integrante debe descargarlo:

1. Entrar a: https://www.kaggle.com/datasets/luiszhinin979/ecuadorian-cocoa-beans-for-classification
2. Descargar el `.zip` (requiere cuenta de Kaggle).
3. Extraerlo dentro de `data/raw/`, de modo que queden estas carpetas:

```
data/raw/
├── dataset_cropped_allclasses/        # 1634 recortes del dataset (NO se usan para entrenar)
├── dataset_original_train_val/        # fotos originales (train/ y val/): fuente de nuestros recortes
└── labels_croppedallclasses.json      # anotaciones de recorte (no se usan)
```

`data/raw/` está en el `.gitignore`: nunca se sube ni se modifica.

## Recortes propios (datos de entrenamiento)

Los modelos se entrenan con recortes generados por el equipo a partir de las fotos originales de `dataset_original_train_val/`, con el script `generar_recortes.py` (OpenCV, sin filtros sobre la imagen). El resultado está en la carpeta `recortes_script/`, que se copia a `data/processed/recortes_script/` (ignorada por Git).

- `train/`: 1300 recortes (1000 sana, 300 mala) y `val/`: 334 recortes (261 sana, 73 mala).
- **Verificado con `04_inventory_crops.py`:** conteos por clase iguales a los del LEEME, 0 archivos corruptos, 0 duplicados por md5, 0 contenido compartido entre train y val, y los logs coinciden con lo que hay en disco.
- La clase sale del nombre del archivo (`sana (N).jpg`, `mala (N).jpg`).
- Varios nombres de `val/` repiten los de `train/` (son granos distintos): por eso los CSV identifican cada recorte por su ruta completa.

## Estado actual

- [x] Estructura del repositorio
- [x] Inventario y auditoría del dataset original
- [x] Respuestas del profesor (ver "Decisiones acordadas")
- [x] Verificación de los recortes propios (conteos, integridad, duplicados)
- [x] Partición sobre los recortes propios (ver "Partición")
- [ ] Confirmar la partición con el profesor
- [ ] Conteos y gráfico de la distribución por partición
- [ ] Entorno con TensorFlow (pendiente: WSL2 o Colab/Kaggle)
- [ ] Preprocesamiento y data generator, GAN
- [ ] 5 modelos (3 CNN y 2 Transformers) con Federated Learning, hyperparameter tuning, validación cruzada, ensembles
- [ ] Back-end, front-end, pruebas, deployment, monitoreo

## Estructura

```
data/
├── raw/        # dataset original de Kaggle (NO se versiona)
├── processed/  # recortes propios y datos procesados (NO se versionan)
└── splits/     # train.csv, val.csv, test.csv (SÍ se versionan)
ml/
├── src/        # scripts numerados en orden de ejecución
└── results/    # inventory.csv, inventory_crops.csv y resultados
```

## Dataset: hallazgos de la auditoría

- `data/raw/dataset_cropped_allclasses/`: 1634 imágenes (una por grano), etiqueta en el nombre (`sana (N).jpg`, `mala (N).jpg`).
- `data/raw/dataset_original_train_val/`: 1634 fotos de cámara (4624x3472) ya divididas en train (1300) y val (334).
- `labels_croppedallclasses.json`: anotaciones de recorte (formato VGG Image Annotator); no se usa para clasificar.
- 0 imágenes corruptas y 0 duplicados exactos.
- Las carpetas `train/val` y `cropped` no comparten contenido idéntico. Hipótesis (no confirmada al 100%): son las fotos originales y sus recortes de los mismos granos. En `val`, los nombres de las 73 malas se repiten con los de `train` (contenido distinto), por lo que el nombre no sirve como clave de emparejamiento.

**Decisión (18-oct-2026):** `dataset_cropped_allclasses` no se usa para entrenar: trae un filtro que la app no aplica y no se sabe a qué foto original corresponde cada recorte. El equipo generó sus propios recortes a partir de las fotos originales con `generar_recortes.py`.

## Partición (semilla 42, estratificada)

Partimos de los recortes propios, que ya vienen separados por el equipo en `train` (1300) y `val` (334). La partición se hace así (`05_split.py`):

| Conjunto | Origen | Uso | Recortes | sana | mala | % del total |
|---|---|---|---|---|---|---|
| train | 85 % de `train` del equipo | entrenar (y repartir entre clientes de Federated Learning) | 1105 | 850 | 255 | 67,6 % |
| val | 15 % de `train` del equipo | elegir modelo, hiperparámetros, early stopping | 195 | 150 | 45 | 11,9 % |
| test | `val` del equipo, sin tocar | medición final del F1, usada una sola vez | 334 | 261 | 73 | 20,4 % |

Cada CSV de `data/splits/` trae las columnas `origen`, `filename`, `path`, `label` y `md5`. Ojo con los nombres: lo que el documento de requerimientos llama `val` (334) aquí es `test`.

### Justificación frente al 70/15/15 pedido

El profesor pidió una partición 70/15/15. El resultado es ≈ 68/12/20, es decir, **no es exacta**. Se adoptó porque cumple el propósito de la pauta:

- Tres conjuntos disjuntos (sin solapamiento por ruta ni por md5, verificado con `assert` en el script) y estratificados por clase (≈ 23 % `mala` en cada uno).
- Sin data leakage: el `val` del equipo se conserva intacto como test y nunca se usa para entrenar ni para elegir modelo.
- Separa la selección de modelo (`val`) de la medición final (`test`), de modo que el F1 reportado no queda optimista.
- Un test de 334 recortes (73 `mala`) es más grande y estable que uno de 246 (56 `mala`), y respeta la definición de `val` del documento de requerimientos (D-18, RNF-07, RF-08) y del LEEME de los recortes, sin cambiarlos.

**Pendiente:** confirmar esta justificación con el profesor. Si pide 70/15/15 exacto, solo hay que cambiar `05_split.py` (juntar los 1634 y repartir) y ajustar D-18 y RNF-07.

## Cómo reproducir

1. Descargar y extraer el dataset en `data/raw/` (ver "Descarga del dataset").
2. Copiar `recortes_script/` a `data/processed/recortes_script/`.
3. Instalar dependencias: `pip install pandas pillow scikit-learn`
4. Desde la raíz del proyecto, ejecutar en orden:

```
python ml/src/00_inventory.py        # inventario del dataset original -> ml/results/inventory.csv
python ml/src/01_check_pairs.py      # verifica relación cropped vs train/val
python ml/src/02_check_names.py      # revisa nombres repetidos
python ml/src/04_inventory_crops.py  # inventario de los recortes propios -> ml/results/inventory_crops.csv
python ml/src/05_split.py            # partición -> data/splits/*.csv
```

(Los scripts 00 a 02 documentan la auditoría que llevó a descartar `cropped`. El 03 se eliminó: era la partición sobre `cropped`.)

## Decisiones acordadas

1. Framework: TensorFlow (obligatorio, el profesor no permite PyTorch).
2. Se usará Federated Learning (más simple, porque una sola persona entrena los modelos).
3. Además de los CSV de partición, se guardan conteos por clase y un gráfico.
4. El dataset va solo como enlace, no dentro del repo.
5. Se entrenan 5 modelos (3 CNN y 2 Transformers) con recortes propios, no con `cropped`.
6. Partición sobre recortes propios: 85/15 del train del equipo + val del equipo como test (pendiente de confirmar con el profesor).

## Pendientes

- Confirmar la partición ≈ 68/12/20 con el profesor.
- Generar y guardar los conteos por clase y el gráfico de la distribución.
- Fijar la métrica y el valor mínimo (el equipo habla de 94,5 % o 95 %: aclarar si es F1 macro o exactitud).
- Definir el entorno de entrenamiento con TensorFlow (WSL2 o Colab/Kaggle).
- Definir cuántos clientes simulados tendrá el Federated Learning y cómo se reparte `train` entre ellos.