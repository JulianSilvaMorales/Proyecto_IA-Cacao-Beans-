"""
Nombre: 08_check_models.py
Descripcion: Comprueba que los 5 presets elegidos (3 CNN y 2 Transformers)
             descargan y cargan con 2 clases de salida, y mide el tiempo de
             inferencia por imagen en GPU y CPU (lote de 32, 224x224). No entrena.
             La CPU es la de este equipo, no la t4g.medium de la app: sirve para
             comparar los modelos entre si, no como cifra final.
Fecha_creacion: 2026-10-19
Actualizacion: 2026-10-19
Autor: Julian Andres Silvas Morales
Version: 1.1
"""
import gc
import os
import sys
import time

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import keras_hub as kh
import numpy as np
import tensorflow as tf

SEED = 42
BATCH = 32
IMG = 224
REPS = 3
keras.utils.set_random_seed(SEED)

MODELOS = [
    ("MobileNetV3-L", kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    ("EfficientNet-B0", kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    ("ResNet-50", kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    ("Swin-Tiny", kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    ("DeiT-Small", kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
]

if len(sys.argv) > 1:   # opcional: correr solo los modelos indicados por nombre
    MODELOS = [m for m in MODELOS if m[0] in sys.argv[1:]]

x = np.random.default_rng(SEED).uniform(0, 255, (BATCH, IMG, IMG, 3)).astype("float32")
xt = tf.convert_to_tensor(x)


def medir(cls, preset, device):
    """Carga el modelo en 'device' y devuelve (params, forma_salida, ms/imagen)."""
    with tf.device(device):
        model = cls.from_preset(preset, num_classes=2, preprocessor=None)
        for _ in range(3):
            model(xt, training=False).numpy()          # calentamiento
        t0 = time.perf_counter()
        for _ in range(REPS):
            out = model(xt, training=False).numpy()    # .numpy() espera a la GPU
        ms = (time.perf_counter() - t0) / REPS / BATCH * 1000
    n = model.count_params()
    del model
    keras.backend.clear_session()
    gc.collect()
    return n, out.shape, ms


print(f"GPUs visibles: {tf.config.list_physical_devices('GPU')}")
print(f"Lote: {BATCH} imagenes de {IMG}x{IMG}, {REPS} repeticiones\n")
print(f"{'modelo':16} {'params (M)':>10} {'salida':>10} {'GPU ms/img':>11} {'CPU ms/img':>11}")
for nombre, cls, preset in MODELOS:
    try:
        p, shape, ms_gpu = medir(cls, preset, "/GPU:0")
        _, _, ms_cpu = medir(cls, preset, "/CPU:0")
        print(f"{nombre:16} {p/1e6:10.1f} {str(shape):>10} {ms_gpu:11.2f} {ms_cpu:11.2f}")
    except Exception as e:
        print(f"{nombre:16} FALLA -> {type(e).__name__}: {str(e)[:200]}")
