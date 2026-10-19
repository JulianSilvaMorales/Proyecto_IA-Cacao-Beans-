# Clasificación de granos de cacao (Proyecto final IA): InspectIA

Universidad Surcolombiana · Ingeniería de Software · Curso de Inteligencia Artificial.
Docente: Juan Antonio Castro Silva.

Dataset: "Ecuadorian cacao beans for classification" (Kaggle).
1634 granos únicos: 1261 `sana` y 373 `mala` (enfermos). Recolectados en La Maná, Ecuador.

Licencia del dataset: CC BY-NC 4.0 (uso académico citando a los autores; no permite uso comercial).
Fuente: Zhinin-Vera, L. et al. (2023). Artificial Vision Technique to Detect and Classify Cocoa Beans. IWANN 2023, LNCS 14135, Springer. DOI 10.1007/978-3-031-43078-7_18.

## Estado actual

- [x] Estructura del repositorio
- [x] Inventario y auditoría del dataset original
- [x] Respuestas del profesor (ver "Decisiones acordadas")
- [x] Verificación de los recortes propios (conteos, integridad, duplicados)
- [x] Partición sobre los recortes propios
- [x] Conteos y gráfico de la distribución por partición
- [x] Entorno de entrenamiento: WSL2 + Ubuntu 24.04 + TensorFlow 2.21.0 con GPU verificada
- [x] Repo clonado en Ubuntu y pipeline de datos reproducido allí (misma partición y mismos conteos)
- [x] Data generator (pipeline `tf.data`), línea base
- [ ] Confirmar la partición con el profesor
- [ ] GAN para la clase minoritaria (solo `train`)
- [ ] Definir los 5 modelos (3 CNN y 2 Transformers) y los clientes de Federated Learning
- [ ] Entrenamiento, hyperparameter tuning, validación cruzada, ensembles
- [ ] Back-end, front-end, pruebas, deployment, monitoreo

## Descarga del dataset

El dataset **no se versiona** en el repositorio por su tamaño (el .zip pesa 5,15 GB); el profesor confirmó que basta el enlace. Cada integrante debe descargarlo:

1. Entrar a: https://www.kaggle.com/datasets/luiszhinin979/ecuadorian-cocoa-beans-for-classification
2. Descargar el `.zip` (requiere cuenta de Kaggle).
3. Extraerlo dentro de `data/raw/` (ver la estructura de carpetas más abajo).

## Estructura de carpetas

Lo que está en Git y lo que cada integrante debe tener en local:

```
Proyecto_IA(Cacao-Beans)/             (en Ubuntu: ~/cacao/)
├── README.md
├── .gitignore
├── data/
│   ├── raw/                          NO versionado (solo .gitkeep)
│   │   ├── .gitkeep
│   │   ├── dataset_cropped_allclasses/        1634 .jpg (NO se usan para entrenar)
│   │   ├── dataset_original_train_val/
│   │   │   ├── train/                         1300 fotos de cámara
│   │   │   └── val/                           334 fotos de cámara
│   │   └── labels_croppedallclasses.json      anotaciones (no se usan)
│   ├── processed/                    NO versionado (solo .gitkeep)
│   │   ├── .gitkeep
│   │   └── recortes_script/                   recortes propios del equipo
│   │       ├── train/                         1300 .jpg (1000 sana, 300 mala)
│   │       ├── val/                           334 .jpg (261 sana, 73 mala)
│   │       ├── generar_recortes.py            script que generó los recortes
│   │       ├── LEEME.txt
│   │       ├── log_train.csv
│   │       ├── log_val.csv
│   │       └── muestra_recortes.jpg
│   └── splits/                       SÍ versionado
│       ├── train.csv                          1105 filas
│       ├── val.csv                            195 filas
│       └── test.csv                           334 filas
└── ml/
    ├── src/                          scripts numerados (ver "Cómo reproducir")
    └── results/                      resultados, SÍ versionado
        ├── inventory.csv
        ├── inventory_crops.csv
        ├── split_counts.csv
        ├── split_distribution.png
        └── pipeline_sample.png
```

Cada CSV de `data/splits/` trae las columnas `origen`, `filename`, `path`, `label` y `md5`. La columna `path` es **relativa a la raíz del repo** (por ejemplo `data/processed/recortes_script/train/mala (1).jpg`), así que funciona igual en Windows y en Ubuntu si los scripts se ejecutan desde la raíz.

## Recortes propios (datos de entrenamiento)

Los modelos se entrenan con recortes generados por el equipo a partir de las fotos originales de `dataset_original_train_val/`, con el script `generar_recortes.py` (OpenCV, sin filtros sobre la imagen). Ese script **no está versionado en este repo**: viene dentro de la carpeta `recortes_script/` que entrega el equipo (ver su `LEEME.txt` para el procedimiento). No se usa `dataset_cropped_allclasses`: trae un filtro que la app no aplica y no se sabe a qué foto original corresponde cada recorte.

- `train/`: 1300 recortes (1000 sana, 300 mala) y `val/`: 334 recortes (261 sana, 73 mala).
- **Verificado con `04_inventory_crops.py`:** conteos por clase iguales a los del LEEME, 0 archivos corruptos, 0 duplicados por md5, 0 contenido compartido entre train y val, y los logs coinciden con lo que hay en disco.
- Tamaño: mediana 596x370 px, mínimo 340x201, máximo 1459x896.
- La clase sale del nombre del archivo (`sana (N).jpg`, `mala (N).jpg`).
- Varios nombres de `val/` repiten los de `train/` (son granos distintos): por eso los CSV identifican cada recorte por su ruta completa.

## Partición (semilla 42, estratificada)

Los recortes ya vienen separados por el equipo en `train` (1300) y `val` (334). La partición (`05_split.py`) es:

| Conjunto | Origen | Uso | Recortes | sana | mala | % mala | % del total |
|---|---|---|---|---|---|---|---|
| train | 85 % de `train` del equipo | entrenar (y repartir entre clientes de Federated Learning) | 1105 | 850 | 255 | 23,1 | 67,6 |
| val | 15 % de `train` del equipo | elegir modelo, hiperparámetros, early stopping | 195 | 150 | 45 | 23,1 | 11,9 |
| test | `val` del equipo, sin tocar | medición final del F1, usada una sola vez | 334 | 261 | 73 | 21,9 | 20,4 |

Los conteos salen de `06_counts_plot.py` (`ml/results/split_counts.csv`):

![Distribución de clases por partición](ml/results/split_distribution.png)

Ojo con los nombres: lo que el documento de requerimientos llama `val` (334) aquí es `test`.

### Por qué se particionó así

El profesor pidió una partición 70/15/15. El resultado es ≈ 68/12/20, es decir, **no es exacta**. Se eligió porque cumple el propósito de la pauta y da más confianza en la métrica final:

- **Sin data leakage.** Los tres conjuntos son disjuntos: el script verifica con `assert` que no hay solapamiento por ruta ni por md5, y que suman 1634.
- **La selección y la medición final están separadas.** Los 5 modelos se comparan y ajustan con `val`; el F1 que se reporta sale de `test`, que no interviene en ninguna decisión. Si un mismo conjunto eligiera el modelo y además midiera el F1, la cifra saldría optimista.
- **`train` y `val` están estratificados** (23,1 % de `mala` en ambos). `test` es el `val` del equipo, sin tocar, y tiene 21,9 % de `mala`: 1,2 puntos menos, una diferencia pequeña pero real.
- **Coherencia con el documento de requerimientos** (D-18, RNF-07, RF-08) y con el LEEME de los recortes, que definen ese conjunto de 334 como el de medición final. Así no hay que reescribir esas decisiones.
- **Un test de 334 es más grande que uno de 246** (el que saldría con 15 % de 1634).

Limitación: con solo 73 granos `mala` en `test`, un error mueve el recall de esa clase más de un punto. Hay que tenerlo presente al comparar el resultado con la meta del equipo.

**Pendiente:** confirmar esta justificación con el profesor. Si exige 70/15/15 exacto, solo cambia `05_split.py` (juntar los 1634 y repartir) y hay que ajustar D-18 y RNF-07.

## Pipeline de datos (data generator)

`ml/src/07_data_pipeline.py` lee `data/splits/*.csv` y entrega los recortes por lotes con `tf.data`, sin cargar todo en memoria. Es la **línea base**: los parámetros se pueden cambiar y se comparan con `val` cuando haya modelos.

| Decisión | Valor | Por qué |
|---|---|---|
| Tamaño de entrada | 224x224, configurable (`IMG_SIZE`) | Estándar para CNN y ViT; probar 384 más adelante (más detalle, pero más costo, y la app corre sin GPU) |
| Redimensionado | Cuadrado sin deformar el grano, prolongando y suavizando el borde de cada recorte | Estirar deformaría el grano; ver nota de abajo |
| Normalización | Ninguna en el pipeline (píxeles 0 a 255) | Cada modelo preentrenado exige su propio preprocesamiento y se aplica al armarlo |
| Augmentation | Solo en `train`: volteo horizontal y vertical, brillo y contraste leves | `val` y `test` no se modifican; sin cambios de color porque el color distingue `sana` de `mala` (hipótesis, no verificada) |
| Etiquetas | sana = 0, mala = 1 | |
| Lote | 32 | 35 lotes en train, 7 en val, 11 en test |

**Nota sobre el relleno.** Para volver cuadrado un recorte hay que añadir franjas. Con relleno blanco se veía un borde marcado contra el papel. Se midió el color del borde de 200 recortes de `train`: media (165, 165, 167), mínimo 121 y máximo 185 por canal, es decir un gris medio y no blanco. Además, la proporción ancho/alto difiere un poco entre clases (media 1,55 en `mala` y 1,62 en `sana`) y un clasificador que usa solo esa proporción logra F1 macro de 0,607 en `val` (adivinar daría cerca de 0,43). Como el relleno depende de la proporción, podría actuar como atajo. Por eso se prolonga el borde de cada recorte, suavizado a 16 valores, para que la franja se mezcle con el fondo. No está medido contra el relleno blanco: es una comparación pendiente, con `val`.

![Muestra del pipeline](ml/results/pipeline_sample.png)

## Entorno de entrenamiento: WSL2 + TensorFlow con GPU

TensorFlow es obligatorio (decisión del profesor). La documentación oficial indica que **desde la versión 2.11 TensorFlow no tiene soporte de GPU en Windows nativo**; para usar la GPU hay que usar WSL2. Por eso el entrenamiento se hace en Ubuntu dentro de WSL2, y el entorno `.venv` de Windows solo se usó para los scripts de datos 00 a 06.

Equipo de desarrollo: Windows 10 22H2, GPU NVIDIA RTX 3060 Ti (8 GB), driver 596.49.

### Versiones verificadas

| Componente | Versión |
|---|---|
| Sistema | WSL2, Ubuntu 24.04 |
| Python | 3.12.3 (viene con Ubuntu 24.04) |
| TensorFlow | 2.21.0 (`tensorflow[and-cuda]`) |
| pandas / pillow / scikit-learn / matplotlib | 3.0.6 / 13.0.0 / 1.9.1 / 3.11.2 |
| Entorno virtual | `~/entornos/tf` |

Python 3.12 figura en las versiones soportadas por TensorFlow según la documentación oficial. Python 3.14 no aparece, y por eso no se usa para entrenar.

### Paso a paso

**1. Comprobar WSL2 (en CMD de Windows).** Se necesita Windows 10 versión 21H2 o superior y un driver NVIDIA reciente.

```cmd
wsl --status
wsl -l -v
nvidia-smi
```

**2. Instalar Ubuntu 24.04** (en CMD). Pide crear un usuario y una contraseña de Linux.

```cmd
wsl --install Ubuntu-24.04
```

**3. Comprobar que Ubuntu ve la GPU y la versión de Python** (en Ubuntu):

```bash
python3 --version
nvidia-smi
```

**4. Crear el entorno virtual e instalar TensorFlow con GPU:**

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
cd ~
mkdir -p entornos
python3 -m venv entornos/tf
source entornos/tf/bin/activate
pip install --upgrade pip
pip install 'tensorflow[and-cuda]'
```

**5. Indicar dónde están las bibliotecas NVIDIA.** Tras instalar, `tf.config.list_physical_devices('GPU')` devolvía `[]` con el aviso "Cannot dlopen some GPU libraries". Se comprobó que las bibliotecas existían y cargaban por ruta completa, pero **no por nombre**: el sistema no buscaba en las carpetas de pip. Los enlaces simbólicos que sugiere la documentación no resolvieron el problema; lo que funcionó fue añadir esas rutas a `LD_LIBRARY_PATH`. Para que se aplique cada vez que se activa el entorno:

```bash
cat >> ~/entornos/tf/bin/activate <<'EOF'

# Rutas de las bibliotecas NVIDIA instaladas por pip (para que TensorFlow vea la GPU)
_NV_LIBS=$(python -c "import site,glob; print(':'.join(sorted(glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib'))))")
export LD_LIBRARY_PATH="$_NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EOF
```

**6. Instalar las librerías de datos** (en el mismo entorno):

```bash
pip install pandas pillow scikit-learn matplotlib
```

**7. Verificar** (en una terminal nueva, con el entorno activado):

```bash
source ~/entornos/tf/bin/activate
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))" 2>/dev/null
```

Salida esperada: `2.21.0 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.

### Clonar el repo en Ubuntu y copiar los recortes

El repo se trabaja en el disco de Ubuntu (`~/cacao`), no en `/mnt/c`: leer miles de imágenes por `/mnt/c` es mucho más lento. Los recortes no están en Git, así que se copian una sola vez desde Windows:

```bash
cd ~
git clone https://github.com/JulianSilvaMorales/Proyecto_IA-Cacao-Beans-.git cacao
cp -r "/mnt/c/Users/USER/Documents/Proyecto_IA(Cacao-Beans)/data/processed/recortes_script" ~/cacao/data/processed/
cd ~/cacao
git config --global user.name "su-usuario-de-github"
git config --global user.email "su-correo-de-github"
```

(Ajustar la ruta de Windows a cada equipo.) Al hacer `git push` desde Ubuntu, GitHub pide un token de acceso personal en lugar de la contraseña.

**Verificado en Ubuntu:** `04_inventory_crops.py` y `05_split.py` dan los mismos conteos que en Windows (1000/300 y 261/73 en los recortes; 1105/195/334 en la partición), `git status` queda limpio (los CSV regenerados son idénticos a los versionados) y `06_counts_plot.py` da la misma tabla.

### Notas

- Plan B si WSL2 diera problemas: Google Colab o Kaggle (hay que subir los recortes aparte y hacer configurable la ruta base de las imágenes).

## Cómo reproducir

1. Descargar y extraer el dataset en `data/raw/` (ver "Descarga del dataset").
2. Copiar `recortes_script/` a `data/processed/recortes_script/`.
3. Crear el entorno con TensorFlow (ver "Entorno de entrenamiento"). Los scripts 04 a 07 se corrieron en Ubuntu; los 00 a 06 también funcionan en Windows con `pip install pandas pillow scikit-learn matplotlib`.
4. Desde la raíz del proyecto, ejecutar en orden:

```
python ml/src/00_inventory.py        # inventario del dataset original -> ml/results/inventory.csv
python ml/src/01_check_pairs.py      # verifica relación cropped vs train/val
python ml/src/02_check_names.py      # revisa nombres repetidos
python ml/src/04_inventory_crops.py  # inventario de los recortes propios -> ml/results/inventory_crops.csv
python ml/src/05_split.py            # partición -> data/splits/*.csv
python ml/src/06_counts_plot.py      # conteos y gráfico -> ml/results/
TF_CPP_MIN_LOG_LEVEL=2 python ml/src/07_data_pipeline.py   # pipeline tf.data (requiere TensorFlow) -> ml/results/pipeline_sample.png
```

Los scripts 00 a 02 documentan la auditoría que llevó a descartar `dataset_cropped_allclasses`; solo corren con `data/raw/` completo. Del 04 en adelante solo hacen falta los recortes y los CSV. El 03 se eliminó (era la partición sobre `cropped`).

## Hallazgos de la auditoría del dataset original

- `dataset_cropped_allclasses/`: 1634 imágenes (una por grano), etiqueta en el nombre. 0 corruptas y 0 duplicados exactos.
- `dataset_original_train_val/`: 1634 fotos de cámara (4624x3472 y 4624x2136) ya divididas en train (1300) y val (334).
- `labels_croppedallclasses.json`: anotaciones de recorte (formato VGG Image Annotator); no se usa para clasificar.
- Las carpetas `train/val` y `cropped` no comparten contenido idéntico. Hipótesis (no confirmada al 100 %): son las fotos originales y sus recortes de los mismos granos. En `val`, los nombres de las 73 malas se repiten con los de `train` (contenido distinto), por lo que el nombre no sirve como clave de emparejamiento.

## Decisiones acordadas

1. Framework: TensorFlow (obligatorio, el profesor no permite PyTorch).
2. Se usará Federated Learning (más simple, porque una sola persona entrena los modelos).
3. Además de los CSV de partición, se guardan conteos por clase y un gráfico.
4. El dataset va solo como enlace, no dentro del repo.
5. Se entrenan 5 modelos (3 CNN y 2 Transformers) con recortes propios, no con `cropped`.
6. Partición sobre recortes propios: 85/15 del train del equipo + val del equipo como test (pendiente de confirmar con el profesor).
7. Entrenamiento en WSL2 (Ubuntu 24.04) con TensorFlow 2.21.0 y la GPU local.
8. Pipeline de datos de línea base: 224x224 configurable, relleno por borde suavizado, augmentation solo en `train` y sin cambios de color. Tamaño, relleno y augmentation se comparan con `val` más adelante.
9. Meta de trabajo: F1 macro y exactitud de 95 %, medidas en `test`. Aún no está registrada en el documento de requerimientos (P-02). Con solo 73 `mala` en `test`, la cifra tiene un margen de incertidumbre amplio.

## Pendientes

- Confirmar la partición ≈ 68/12/20 con el profesor.
- Definir las 5 arquitecturas (3 CNN y 2 Transformers).
- Definir cuántos clientes simulados tendrá el Federated Learning y cómo se reparte `train` entre ellos.
- GAN para la clase minoritaria (solo `train`).