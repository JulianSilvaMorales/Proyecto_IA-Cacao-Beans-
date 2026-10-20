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
- [x] Partición sobre los recortes propios, conteos y gráfico
- [x] Entorno de entrenamiento: WSL2 + Ubuntu 24.04 + TensorFlow 2.21.0 con GPU verificada
- [x] Data generator (pipeline `tf.data`), línea base
- [x] Elección de las 5 arquitecturas (3 CNN y 2 Transformers) y comprobación de que cargan
- [x] Reparto de `train` entre 5 clientes de Federated Learning
- [x] Ciclo de Federated Learning (FedAvg propio) con checkpoints; probado con un modelo
- [ ] Confirmar la partición con el profesor
- [ ] Entrenamiento formal de los 5 modelos con Federated Learning
- [ ] Hyperparameter tuning, validación cruzada, ensembles, GAN
- [ ] Medición final en `test` (una sola vez)
- [ ] Back-end, front-end, pruebas, deployment, monitoreo

**Meta de trabajo:** F1 macro y exactitud de 95 %, medidas en `test`. **Todavía no se ha alcanzado ni hay evidencia de que sea alcanzable con estos recortes** (ver "Resultados de diagnóstico").

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
│       ├── test.csv                           334 filas
│       └── federated/                         reparto de train entre clientes
│           └── client_0.csv ... client_4.csv  221 filas cada uno
└── ml/
    ├── src/                          scripts numerados (ver "Cómo reproducir")
    └── results/                      resultados, SÍ versionado
        ├── inventory.csv, inventory_crops.csv
        ├── split_counts.csv, split_distribution.png
        ├── pipeline_sample.png
        ├── models_check.txt
        ├── federated_counts.csv
        └── errors_val.png
```

Fuera del repo (no se versiona): `~/cacao_runs/` guarda los checkpoints y el historial de cada entrenamiento, y `~/.cache/kagglehub/` guarda los pesos preentrenados descargados.

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
- **La selección y la medición final están separadas.** Los modelos se comparan y ajustan con `val`; el F1 que se reporta sale de `test`, que no interviene en ninguna decisión. Si un mismo conjunto eligiera el modelo y además midiera el F1, la cifra saldría optimista.
- **`train` y `val` están estratificados** (23,1 % de `mala` en ambos). `test` es el `val` del equipo, sin tocar, y tiene 21,9 % de `mala`: 1,2 puntos menos, una diferencia pequeña pero real.
- **Coherencia con el documento de requerimientos** (D-18, RNF-07, RF-08) y con el LEEME de los recortes, que definen ese conjunto de 334 como el de medición final.
- **Un test de 334 es más grande que uno de 246** (el que saldría con 15 % de 1634).

Limitación: con solo 73 granos `mala` en `test` (y 45 en `val`), un error mueve el recall de esa clase más de un punto. Hay que tenerlo presente al comparar resultados con la meta.

**Pendiente:** confirmar esta justificación con el profesor. Si exige 70/15/15 exacto, solo cambia `05_split.py` (juntar los 1634 y repartir) y hay que ajustar D-18 y RNF-07.

## Pipeline de datos (data generator)

`ml/src/07_data_pipeline.py` lee `data/splits/*.csv` y entrega los recortes por lotes con `tf.data`, sin cargar todo en memoria. Es la **línea base**: los parámetros se pueden cambiar y se comparan con `val`.

| Decisión | Valor | Por qué |
|---|---|---|
| Tamaño de entrada | 224x224, configurable (`IMG_SIZE`) | Todos los presets elegidos son de 224; probar 384 más adelante |
| Redimensionado | Cuadrado sin deformar el grano, prolongando y suavizando el borde de cada recorte | Estirar deformaría el grano; ver nota de abajo |
| Normalización | Ninguna en el pipeline (píxeles 0 a 255) | Cada modelo preentrenado aplica la suya (ver "Lecciones del entrenamiento") |
| Augmentation | Solo en `train`: volteo horizontal y vertical, brillo y contraste leves | `val` y `test` no se modifican; sin cambios de color porque el color distingue `sana` de `mala` (hipótesis, no verificada) |
| Etiquetas | sana = 0, mala = 1 | |
| Lote | 32 | 35 lotes en train, 7 en val, 11 en test |

**Nota sobre el relleno.** Para volver cuadrado un recorte hay que añadir franjas. Con relleno blanco se veía un borde marcado contra el papel. Se midió el color del borde de 200 recortes de `train`: media (165, 165, 167), mínimo 121 y máximo 185 por canal, es decir un gris medio y no blanco. Además, la proporción ancho/alto difiere un poco entre clases (media 1,55 en `mala` y 1,62 en `sana`) y un clasificador que usa solo esa proporción logra F1 macro de 0,607 en `val` (0,543 en validación cruzada). Como el relleno depende de la proporción, podría actuar como atajo. Por eso se prolonga el borde de cada recorte, suavizado a 16 valores. No está medido contra el relleno blanco: es una comparación pendiente, con `val`.

![Muestra del pipeline](ml/results/pipeline_sample.png)

## Entorno de entrenamiento: WSL2 + TensorFlow con GPU

TensorFlow es obligatorio (decisión del profesor). La documentación oficial indica que **desde la versión 2.11 TensorFlow no tiene soporte de GPU en Windows nativo**; para usar la GPU hay que usar WSL2. Por eso el entrenamiento se hace en Ubuntu dentro de WSL2, y el entorno `.venv` de Windows solo se usó para los scripts de datos 00 a 06.

Equipo de desarrollo: Windows 10 22H2, GPU NVIDIA RTX 3060 Ti (8 GB), driver 596.49. Ubuntu en WSL2 ve unos 15 GiB de RAM.

### Versiones verificadas

| Componente | Versión |
|---|---|
| Sistema | WSL2, Ubuntu 24.04 |
| Python | 3.12.3 (viene con Ubuntu 24.04) |
| TensorFlow / Keras | 2.21.0 (`tensorflow[and-cuda]`) / 3.15.1 |
| keras-hub | 0.32.0 (modelos preentrenados) |
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

**6. Instalar las librerías de datos y de modelos** (en el mismo entorno):

```bash
pip install pandas pillow scikit-learn matplotlib keras-hub
```

`keras-hub` no cambia `tensorflow`, `keras` ni `numpy` (se comprobó con `pip install --dry-run` antes). Al instalarlo, la GPU siguió visible.

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

(Ajustar la ruta de Windows a cada equipo.) Al hacer `git push` desde Ubuntu, GitHub pide un token de acceso personal en lugar de la contraseña (no se guarda ni se comparte).

**Verificado en Ubuntu:** `04_inventory_crops.py`, `05_split.py` y `06_counts_plot.py` dan los mismos resultados que en Windows y `git status` queda limpio.

### Entrenamientos largos en WSL2

- **Al cerrar la última ventana de Ubuntu, WSL2 apaga la distribución y mata los procesos**, incluido `tmux` (se probó: el proceso murió a los ~78 s).
- Lo que funcionó: dejar abierta (puede estar minimizada) una ventana de CMD con `wsl -d Ubuntu-24.04 -- sleep infinity`. Con ella, `tmux` sobrevivió al cierre de la ventana de Ubuntu. Se probó una sola vez y en una prueba corta.
- `tmux` abre una shell nueva que no hereda el entorno: dentro hay que volver a ejecutar `source ~/entornos/tf/bin/activate`.
- En la práctica se entrenó en primer plano, con `2>&1 | tee archivo.log`. El script guarda un checkpoint por ronda en `~/cacao_runs/` y, si se corta, el mismo comando reanuda desde la última ronda.
- Antes de un entrenamiento largo, configurar Windows para no suspenderse mientras está enchufado.

## Modelos y Federated Learning

### Los 5 modelos (decisión del equipo)

Todos preentrenados en ImageNet (transfer learning): con ~1100 imágenes de entrenamiento no tiene sentido entrenarlos desde cero. Todos de 224x224, así que comparten el mismo pipeline. La verificación (`08_check_models.py`, resultado en `ml/results/models_check.txt`) confirmó que los 5 cargan con 2 clases de salida.

| # | Modelo | Preset de KerasHub | Parámetros | Tipo |
|---|---|---|---|---|
| 1 | MobileNetV3-L | `mobilenet_v3_large_100_imagenet` | 4,2 M | CNN |
| 2 | EfficientNet-B0 | `efficientnet_b0_ra4_e3600_r224_imagenet` | 4,1 M | CNN |
| 3 | ResNet-50 | `resnet_50_imagenet` | 23,6 M | CNN |
| 4 | Swin-Tiny | `swin_tiny_patch4_window7_224` | 27,5 M | Transformer |
| 5 | DeiT-Small | `deit_small_distilled_patch16_224_imagenet` | 21,7 M | Transformer |

Criterios: diseños distintos entre sí (para los ensembles del Topic 5) y tamaño compatible con inferencia en CPU (D-18, la app corre sin GPU).

**Tiempos medidos** (lote de 32, en este equipo; la CPU no es la de la t4g.medium de AWS, así que sirven para comparar modelos entre sí, no como cifra final):

| Modelo | GPU ms/img | CPU ms/img |
|---|---|---|
| MobileNetV3-L | 41,6 | 30,4 |
| EfficientNet-B0 | 4,8 | 21,1 |
| ResNet-50 | 2,7 | 30,6 |
| Swin-Tiny | 7,3 | 55,3 |
| DeiT-Small | 3,7 | 48,9 |

El tiempo de MobileNet en GPU (41,6) es anómalo (más lento que en CPU y no cambió con más calentamiento): no se sabe la causa, y esa medición no es fiable. La medición definitiva de D-18 se hace con el modelo ya entrenado y de preferencia en una CPU parecida a la de AWS.

### Federated Learning (decisión mía, a confirmar con el equipo)

- **5 clientes simulados** con reparto estratificado de `train` (`09_federated_clients.py`): cada uno recibe 221 granos (170 `sana` y 51 `mala`), sin repetir ninguno (verificado). Con 10 clientes cada uno tendría ~25 `mala`, demasiado pocos.
- **FedAvg propio** en Keras (`10_train_federated.py`): en cada ronda cada cliente entrena localmente desde los pesos globales y el servidor promedia los pesos ponderando por tamaño. No se comprobó si TensorFlow Federated funciona con TensorFlow 2.21 y Python 3.12; el profesor solo exige usar Federated Learning, no una librería concreta.
- `val` y `test` son globales: los clientes no los ven. Cada ronda se mide el F1 macro en `val`, se guarda la mejor ronda y un checkpoint para reanudar.
- El reparto estratificado es el caso fácil (todos los clientes tienen datos parecidos).

## Lecciones del entrenamiento

Errores y hallazgos de la etapa de diagnóstico, para no repetirlos:

1. **`predict` y `fit` de los modelos de KerasHub aplican su propio preprocesador.** Si se preprocesan los datos a mano y además se pasan a `predict`/`fit`, el preprocesamiento se aplica dos veces. Se comprobó (diferencia de 0,34 en las probabilidades). Solución: cargar el modelo con `preprocessor=None` y aplicar el `image_converter` del preset una sola vez. Las primeras corridas (`smoke`, `t1`, `t2`, `t3`) se hicieron con este error y no sirven como evidencia.
2. **El preprocesamiento correcto es el del preset.** En un sondeo lineal, el converter del preset dio F1 macro 0,668 en `val`; la normalización de ImageNet, 0,478 (azar); solo dividir entre 255, 0,631.
3. **Con una tasa de aprendizaje de 1e-3 y 3e-4 el entrenamiento divergió** (pérdida de millones), y con 1e-4 fue inestable (saltos entre "todo `mala`" y "todo `sana`"). Con **tasa de 1e-5 a 3e-5, recorte de gradiente (`clipnorm` 1.0) y `BatchNorm` congeladas** el entrenamiento es estable. No se aisló cuál de las tres medidas pesa más.
4. **Las `BatchNorm` congeladas** evitan que las estadísticas de entrenamiento e inferencia difieran (con pocos datos se actualizan despacio). En la prueba centralizada, el F1 de `train` pasó de ~0,5 a ~0,7 con la misma pérdida.
5. **Hay sobreajuste:** en la prueba centralizada de 20 épocas, `train` llegó a F1 ~0,95 y `val` se quedó en ~0,70 sin tendencia. Más épocas solas no lo arreglan.
6. Cada ronda de FedAvg tarda ~90 s con 5 clientes, y casi todo parece ser un costo fijo (probablemente recompilar el modelo para cada cliente, no verificado) que casi no depende de las épocas locales.

## Resultados de diagnóstico (no son el entrenamiento formal)

Todas son con **EfficientNet-B0**, medidas en `val` (195 recortes, 45 `mala`), sin usar `test`. Con tan pocas `mala`, las diferencias pequeñas son ruido.

| Prueba | F1 macro | Comentario |
|---|---|---|
| Azar / siempre `sana` | ~0,43 | exactitud 0,77 |
| Solo proporción ancho/alto (CV 5 particiones sobre train+val) | 0,543 ± 0,020 | |
| Regresión logística sobre características congeladas (CV) | 0,669 ± 0,024 | las características aportan señal sobre la proporción |
| Entrenamiento centralizado estable (20 épocas) | 0,60 a 0,73 | `train` ~0,95: sobreajuste |
| **Federated Learning, 10 rondas x 2 épocas locales, lr 3e-5, clipnorm 1, BatchNorm congeladas** | **0,677 (ronda 10)** | exactitud 0,774, recall `mala` 0,49; estable de la ronda 5 a la 10 |

Lectura honesta:

- **El 95 % no se acerca con este enfoque** y no hay evidencia de que sea alcanzable con estos recortes. No se debe presentar de otra forma.
- El Federated Learning rinde a un nivel parecido al centralizado, así que FedAvg no parece ser el cuello de botella (resultado de una sola arquitectura y un reparto).
- Una exactitud de 0,77 equivale a la de predecir siempre `sana`: la métrica que importa es el F1 macro y el recall de `mala`.
- Posibles causas, **sin comprobar**: la etiqueta se asignó por morfología y fermentación (D-17 del documento ya reconoce que el modelo mira solo el exterior); en una cuadrícula de errores (`ml/results/errors_val.png`) varias `mala` mal clasificadas se parecen a `sana` a simple vista, y algunas `sana` tienen manchas blancas parecidas a las de `mala`. Entre `mala` bien clasificadas se ve más negro, puntos blancos y moho.
- Falta probar ajuste fino en dos fases (cabeza sola con la red congelada y luego descongelar con tasa menor): hay una opción `--head-epochs` en `11_central_check.py`, sin resultado todavía.

## Cómo reproducir

1. Descargar y extraer el dataset en `data/raw/` (ver "Descarga del dataset").
2. Copiar `recortes_script/` a `data/processed/recortes_script/`.
3. Crear el entorno con TensorFlow (ver "Entorno de entrenamiento"). Los scripts 00 a 06 también funcionan en Windows con `pip install pandas pillow scikit-learn matplotlib`.
4. Desde la raíz del proyecto, ejecutar en orden:

```
python ml/src/00_inventory.py        # inventario del dataset original -> ml/results/inventory.csv
python ml/src/01_check_pairs.py      # verifica relación cropped vs train/val
python ml/src/02_check_names.py      # revisa nombres repetidos
python ml/src/04_inventory_crops.py  # inventario de los recortes propios -> ml/results/inventory_crops.csv
python ml/src/05_split.py            # partición -> data/splits/*.csv
python ml/src/06_counts_plot.py      # conteos y gráfico -> ml/results/
TF_CPP_MIN_LOG_LEVEL=2 python ml/src/07_data_pipeline.py   # pipeline tf.data -> ml/results/pipeline_sample.png
python ml/src/08_check_models.py     # carga y tiempo de los 5 presets (descarga pesos la primera vez)
python ml/src/09_federated_clients.py  # reparto de train entre 5 clientes -> data/splits/federated/
```

Entrenamiento federado de un modelo (la configuración estable encontrada hasta ahora):

```
python -u ml/src/10_train_federated.py --model EfficientNet-B0 --rounds 10 --local-epochs 2 \
    --lr 3e-5 --clipnorm 1.0 --freeze-bn --tag f1
```

Modelos válidos para `--model`: `MobileNetV3-L`, `EfficientNet-B0`, `ResNet-50`, `Swin-Tiny`, `DeiT-Small`. Los resultados y checkpoints quedan en `~/cacao_runs/<tag>_<modelo>/`.

Scripts de diagnóstico (no son parte del entrenamiento final): `11_central_check.py` (entrenamiento centralizado con métricas en `train` y `val`), `12_feature_probe.py` y `13_probe_cv.py` (sondeo lineal sobre características congeladas) y `14_errors_view.py` (cuadrícula de aciertos y errores).

Los scripts 00 a 02 documentan la auditoría que llevó a descartar `dataset_cropped_allclasses`; solo corren con `data/raw/` completo. El 03 se eliminó (era la partición sobre `cropped`).

## Hallazgos de la auditoría del dataset original

- `dataset_cropped_allclasses/`: 1634 imágenes (una por grano), etiqueta en el nombre. 0 corruptas y 0 duplicados exactos.
- `dataset_original_train_val/`: 1634 fotos de cámara (4624x3472 y 4624x2136) ya divididas en train (1300) y val (334).
- `labels_croppedallclasses.json`: anotaciones de recorte (formato VGG Image Annotator); no se usa para clasificar.
- Las carpetas `train/val` y `cropped` no comparten contenido idéntico. Hipótesis (no confirmada al 100 %): son las fotos originales y sus recortes de los mismos granos. En `val`, los nombres de las 73 malas se repiten con los de `train` (contenido distinto), por lo que el nombre no sirve como clave de emparejamiento.
- La descripción de Kaggle habla de 3268 imágenes (1634 granos, entero y corte transversal). No se comprobó cómo se relaciona esa cifra con las carpetas descargadas; nosotros usamos una sola imagen por grano.

## Decisiones acordadas

1. Framework: TensorFlow (obligatorio, el profesor no permite PyTorch).
2. Se usará Federated Learning (más simple, porque una sola persona entrena los modelos). El profesor no exige una librería concreta.
3. Además de los CSV de partición, se guardan conteos por clase y un gráfico.
4. El dataset va solo como enlace, no dentro del repo.
5. Se entrenan 5 modelos (3 CNN y 2 Transformers) con recortes propios, no con `cropped`.
6. Partición sobre recortes propios: 85/15 del train del equipo + val del equipo como test (pendiente de confirmar con el profesor).
7. Entrenamiento en WSL2 (Ubuntu 24.04) con TensorFlow 2.21.0 y la GPU local.
8. Pipeline de datos de línea base: 224x224 configurable, relleno por borde suavizado, augmentation solo en `train` y sin cambios de color.
9. Modelos preentrenados (transfer learning) con las 5 arquitecturas de la tabla; 5 clientes de Federated Learning con reparto estratificado y FedAvg propio.
10. Meta de trabajo: F1 macro y exactitud de 95 %, medidas en `test`. Aún no está registrada en el documento de requerimientos (P-02).

## Pendientes

- Entrenar formalmente los 5 modelos con Federated Learning (rondas, épocas locales y tasa de aprendizaje por modelo) y comparar con `val`.
- Hyperparameter tuning (Topic 2), validación cruzada (Topic 3), ensembles (Topic 5) y GAN para la clase minoritaria (solo `train`).
- Probar el ajuste fino en dos fases (`--head-epochs`).
- Comparar con `val` el relleno (blanco contra borde suavizado), el tamaño (224 contra 384) y el augmentation.
- Medir `test` una sola vez al final, y el tiempo por grano en una CPU parecida a la t4g.medium (D-18).
- Confirmar la partición ≈ 68/12/20 con el profesor.
- Pasar a Cristhian el preprocesamiento del pipeline y el `image_converter` de cada modelo: el servicio de IA de la app debe aplicar el mismo (RF-08).
- Guardar los prompts usados con IA en `docs/prompts/`.