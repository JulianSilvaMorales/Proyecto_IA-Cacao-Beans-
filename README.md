# Clasificación de granos de cacao (Proyecto final IA)

Dataset: "Ecuadorian cacao beans for classification" (Kaggle).
1634 granos únicos: 1261 `sana` y 373 `mala` (enfermos). Recolectados en La Maná, Ecuador.

## Descarga del dataset

El dataset **no se versiona** en el repositorio por su tamaño. Cada integrante debe descargarlo:

1. Entrar a: https://www.kaggle.com/datasets/luiszhinin979/ecuadorian-cocoa-beans-for-classification
2. Descargar el `.zip` (requiere cuenta de Kaggle).
3. Extraerlo dentro de `data/raw/`, de modo que queden estas carpetas:

```
data/raw/
├── dataset_cropped_allclasses/        # 1634 recortes (fuente que usamos)
├── dataset_original_train_val/        # fotos originales (train/ y val/), no se usan
└── labels_croppedallclasses.json      # anotaciones de recorte, no se usan
```

`data/raw/` está en el `.gitignore`: nunca se sube ni se modifica.

## Estado actual

- [x] Estructura del repositorio
- [x] Inventario y auditoría del dataset
- [x] Partición estratificada 70/15/15 sin data leakage
- [ ] Preprocesamiento y data generator
- [ ] Modelos, hyperparameter tuning, validación cruzada, ensembles
- [ ] Back-end, front-end, pruebas, deployment, monitoreo

## Estructura

```
data/
├── raw/        # dataset original (NO se versiona)
├── processed/  # datos procesados (NO se versiona)
└── splits/     # train.csv, val.csv, test.csv (SÍ se versionan)
ml/
├── src/        # scripts numerados en orden de ejecución
└── results/    # inventory.csv y resultados
```

## Dataset: hallazgos de la auditoría

- `data/raw/dataset_cropped_allclasses/`: 1634 imágenes (una por grano), etiqueta en el nombre (`sana (N).jpg`, `mala (N).jpg`).
- `data/raw/dataset_original_train_val/`: 1634 fotos de cámara (4624x3472) ya divididas en train (1300) y val (334).
- `labels_croppedallclasses.json`: anotaciones de recorte (formato VGG Image Annotator); no se usa para clasificar.
- 0 imágenes corruptas y 0 duplicados exactos.
- Las carpetas `train/val` y `cropped` no comparten contenido idéntico. Hipótesis: son las fotos originales y sus recortes de los mismos granos. En `val`, los nombres de las 73 malas se repiten con los de `train` (contenido distinto), por lo que el nombre no sirve como clave de emparejamiento.

**Decisión:** usar solo `dataset_cropped_allclasses` como fuente única, para evitar que un mismo grano caiga en dos particiones (data leakage). La partición original train/val no se usa porque no cumple 70/15/15.

## Partición (semilla 42, estratificada)

| Partición | Imágenes | sana | mala |
|---|---|---|---|
| train | 1143 | 882 | 261 |
| val | 245 | 189 | 56 |
| test | 246 | 190 | 56 |

El script verifica por hash md5 que ningún archivo aparece en más de una partición.

## Cómo reproducir

1. Descargar y extraer el dataset en `data/raw/` (ver sección "Descarga del dataset").
2. Instalar dependencias: `pip install pandas pillow scikit-learn`
3. Desde la raíz del proyecto, ejecutar en orden:

```
python ml/src/00_inventory.py     # inventario -> ml/results/inventory.csv
python ml/src/01_check_pairs.py   # verifica relación cropped vs train/val
python ml/src/02_check_names.py   # revisa nombres repetidos
python ml/src/03_split.py         # genera data/splits/*.csv
```

## Preguntas para el profesor

1. ¿Puedo usar PyTorch en lugar de TensorFlow?
2. ¿Los CSV de partición bastan como "guardar distribuciones" o quiere también conteos o gráficos?
