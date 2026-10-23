"""
Nombre: 15_diag_bn_softmax.py
Descripcion: Diagnostico del montaje de los 5 modelos (origen del fallo de
             MobileNetV3-L). Parte A (por defecto): por modelo, cuenta las
             BatchNorm que existen frente a las que alcanzaba la funcion
             congelar_bn antigua (recursiva por .layers), comprueba si la capa
             de salida aplica de verdad activation="softmax" (en softmax de 2
             clases la media de la salida es exactamente 0,5) y que con
             activation=None la salida son logits de 2 columnas. Parte B
             (--entrenar): 70 pasos cortos de MobileNet (BatchNorm como antes y
             todas congeladas) y de EfficientNet-B0 con lr 1e-4, imprimiendo la
             perdida, P(mala) y la magnitud de lo que llega a la cabeza.
             NO usa test, NO guarda pesos.
Fecha_creacion: 2026-10-23
Actualizacion: 2026-10-23
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo):
    python -u ml/src/15_diag_bn_softmax.py                # parte A, ~2 min
    python -u ml/src/15_diag_bn_softmax.py --entrenar     # parte A + B, ~6 min
"""
import argparse
import importlib
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import keras_hub as kh
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # make_dataset, CLASS_TO_ID

SEED = 42
PASOS = 70          # ~2 epocas de train (35 lotes); MobileNet se "moria" en la epoca 2
CADA = 10
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
}
# (modelo, forma de congelar BatchNorm, lr): la 1.a reproduce la cuadricula antigua
CORRIDAS = [
    ("MobileNetV3-L", "recursivo", 3e-5),
    ("MobileNetV3-L", "plano", 3e-5),
    ("EfficientNet-B0", "plano", 1e-4),
]


def congelar_bn_recursivo(layer):
    """Funcion ANTIGUA de 10/11 (hasta v1.5/v1.4): solo baja por el atributo .layers."""
    n = 1 if isinstance(layer, keras.layers.BatchNormalization) else 0
    if n:
        layer.trainable = False
    for sub in getattr(layer, "layers", None) or []:
        n += congelar_bn_recursivo(sub)
    return n


def bn_planas(model):
    """Todas las BatchNormalization del modelo, recorriendo cualquier subcapa."""
    return [l for l in model._flatten_layers(include_self=False, recursive=True)
            if isinstance(l, keras.layers.BatchNormalization)]


def stats(a):
    a = np.asarray(a, np.float64)
    entre = a.reshape(len(a), -1).std(0).mean()
    return (f"forma {tuple(a.shape)} | media {a.mean():.3g} | desv {a.std():.3g} | "
            f"|max| {np.abs(a).max():.3g} | ceros {np.mean(a == 0):.2f} | "
            f"desv entre imagenes {entre:.3g}")


def cargar(nombre, activation):
    cls, preset = MODELOS[nombre]
    _tmp = cls.from_preset(preset, num_classes=2)
    conv = _tmp.preprocessor.image_converter
    del _tmp
    model = cls.from_preset(preset, num_classes=2, activation=activation, preprocessor=None)
    return model, conv


def parte_a(xv):
    print("\n### PARTE A: montaje de los 5 modelos (sin entrenar)")
    print(f"{'modelo':<16} {'BN plano':>8} {'BN recursivo':>12} "
          f"{'act. salida (softmax pedido)':>30} {'media salida':>13} "
          f"{'act. salida (None)':>20} {'logits min/max':>16} {'suma softmax':>13}")
    for nombre in MODELOS:
        keras.backend.clear_session()
        keras.utils.set_random_seed(SEED)
        try:
            m, conv = cargar(nombre, "softmax")
            xc = conv(xv)
            n_plano = len(bn_planas(m))
            n_rec = congelar_bn_recursivo(m)
            act_s = getattr(m.layers[-1], "activation", None)
            media_s = float(m(xc, training=False).numpy().mean())
            del m
            m, conv = cargar(nombre, None)
            act_n = getattr(m.layers[-1], "activation", None)
            out = m(xc, training=False)
            suma = float(tf.nn.softmax(out).numpy().sum(1).mean())
            print(f"{nombre:<16} {n_plano:>8} {n_rec:>12} "
                  f"{getattr(act_s, '__name__', str(act_s)):>30} {media_s:>13.3f} "
                  f"{getattr(act_n, '__name__', str(act_n)):>20} "
                  f"{float(out.numpy().min()):>7.2f}/{float(out.numpy().max()):<7.2f} "
                  f"{suma:>13.3f} | salida {tuple(out.shape)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR en {nombre}: {type(e).__name__}: {e}", flush=True)
    print("Lectura: si se aplica softmax, 'media salida' es 0,500 exacto.")


def parte_b(nombre, modo, lr, xv):
    print("\n" + "=" * 100)
    print(f"### PARTE B: {nombre} | BatchNorm: {modo} | lr {lr:g} | salida con softmax pedido (como antes)")
    print("=" * 100, flush=True)
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED)
    model, conv = cargar(nombre, "softmax")
    todas = bn_planas(model)
    if modo == "recursivo":
        congelar_bn_recursivo(model)
    else:
        for l in todas:
            l.trainable = False
    print(f"BatchNorm existentes: {len(todas)} | congeladas: {sum(not l.trainable for l in todas)}")
    print(f"converter: {type(conv).__name__} | scale {getattr(conv, 'scale', '?')} | "
          f"offset {getattr(conv, 'offset', '?')}")
    xc = conv(xv)
    print("entrada tras el converter:", stats(xc.numpy()))
    subs = [(l, keras.Model(model.inputs, l.output)) for l in model.layers
            if not isinstance(l, keras.layers.InputLayer)]
    penult = subs[-2][1]

    def capas(titulo):
        print(f"-- salida de cada capa superior ({titulo}):")
        for l, sub in subs:
            print(f"  {l.name:<26} {type(l).__name__:<26} {stats(sub(xc, training=False).numpy())}")

    def pred_info():
        p = np.clip(model(xc, training=False).numpy(), 1e-7, 1.0)
        return f"P(mala) media {p[:, 1].mean():.3f} desv {p[:, 1].std():.4f}"

    capas("sin entrenar")
    tr_df = pd.read_csv("data/splits/train.csv")
    n, n_mala = len(tr_df), int((tr_df["label"] == "mala").sum())
    cw = tf.constant([n / (2 * (n - n_mala)), n / (2 * n_mala)], tf.float32)
    ds = pipe.make_dataset("train", training=True).repeat().map(
        lambda x, y: (conv(x), y, tf.gather(cw, y)), num_parallel_calls=tf.data.AUTOTUNE)
    model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0),
                  loss=keras.losses.SparseCategoricalCrossentropy())
    it = iter(ds)
    t0 = time.time()
    for paso in range(1, PASOS + 1):
        x, y, w = next(it)
        loss = model.train_on_batch(x, y, sample_weight=w)
        loss = float(loss[0] if isinstance(loss, (list, tuple)) else loss)
        if paso % CADA == 0 or paso == 1:
            print(f"paso {paso:>3} | perdida {loss:.4g} | {pred_info()} | "
                  f"entrada a la cabeza: {stats(penult(xc, training=False).numpy())} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    capas(f"tras {PASOS} pasos")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrenar", action="store_true", help="ejecuta tambien la parte B")
    args = ap.parse_args()
    keras.utils.set_random_seed(SEED)
    xv, _ = next(iter(pipe.make_dataset("val", training=False)))   # 32 de val, sin augmentation
    print("GPU:", tf.config.list_physical_devices("GPU"))
    parte_a(xv)
    if args.entrenar:
        for nombre, modo, lr in CORRIDAS:
            try:
                parte_b(nombre, modo, lr, xv)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR en {nombre}/{modo}: {type(e).__name__}: {e}", flush=True)
    print("\nFIN del diagnostico")
