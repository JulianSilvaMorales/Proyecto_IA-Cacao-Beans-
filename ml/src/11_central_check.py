"""
Nombre: 11_central_check.py
Descripcion: Comprobacion de diagnostico. Entrena EfficientNet-B0 de forma
             CENTRALIZADA (todo train junto, sin Federated Learning) con el
             mismo pipeline y preprocesamiento, y mide F1 macro y exactitud en
             train (sin augmentation) y en val tras cada epoca. Sirve para
             separar si un mal resultado viene de FedAvg o del resto. NO usa test.
Fecha_creacion: 2026-10-19
Actualizacion: 2026-10-19
Autor: Julian Andres Silvas Morales
Version: 1.3
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
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")

SEED = 42
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--freeze-bn", action="store_true")
ap.add_argument("--clipnorm", type=float, default=None)
ap.add_argument("--head-epochs", type=int, default=0,
                help="epocas iniciales entrenando solo la cabeza (red congelada)")
args = ap.parse_args()
keras.utils.set_random_seed(SEED)

PRESET = "efficientnet_b0_ra4_e3600_r224_imagenet"
cls = kh.models.EfficientNetImageClassifier
_tmp = cls.from_preset(PRESET, num_classes=2)
conv = _tmp.preprocessor.image_converter
del _tmp
model = cls.from_preset(PRESET, num_classes=2, activation="softmax", preprocessor=None)


def congelar_bn(layer):
    n = 1 if isinstance(layer, keras.layers.BatchNormalization) else 0
    if n:
        layer.trainable = False
    for sub in getattr(layer, "layers", None) or []:
        n += congelar_bn(sub)
    return n


if args.freeze_bn:
    print("BatchNorm congeladas:", congelar_bn(model), flush=True)
model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=args.clipnorm),
              loss=keras.losses.SparseCategoricalCrossentropy())

tr_df = pd.read_csv("data/splits/train.csv")
va_df = pd.read_csv("data/splits/val.csv")
n, n_mala = len(tr_df), int((tr_df["label"] == "mala").sum())
cw = tf.constant([n / (2 * (n - n_mala)), n / (2 * n_mala)], tf.float32)
y_tr = tr_df["label"].map(pipe.CLASS_TO_ID).values
y_va = va_df["label"].map(pipe.CLASS_TO_ID).values
AUTO = tf.data.AUTOTUNE

train_ds = pipe.make_dataset("train", training=True).map(
    lambda x, y: (conv(x), y, tf.gather(cw, y)), num_parallel_calls=AUTO)
train_eval = pipe.make_dataset("train", training=False).map(
    lambda x, y: (conv(x), y), num_parallel_calls=AUTO)
val_ds = pipe.make_dataset("val", training=False).map(
    lambda x, y: (conv(x), y), num_parallel_calls=AUTO)


def medir(ds, y):
    pred = model.predict(ds, verbose=0).argmax(1)
    return (f1_score(y, pred, average="macro"), accuracy_score(y, pred),
            recall_score(y, pred, pos_label=1))


print(f"Centralizado: {n} imagenes de train | peso de clase [sana, mala]: "
      f"{[round(float(v), 2) for v in cw.numpy()]} | lr {args.lr}", flush=True)
def compilar(lr):
    model.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=args.clipnorm),
                  loss=keras.losses.SparseCategoricalCrossentropy())


if args.head_epochs > 0:
    # Fase 1: solo la cabeza (la red preentrenada se congela)
    model.backbone.trainable = False
    compilar(args.lr)
    print("Fase 1: solo cabeza, backbone congelado", flush=True)
for e in range(1, args.epochs + 1):
    if args.head_epochs > 0 and e == args.head_epochs + 1:
        # Fase 2: se descongela todo (BatchNorm sigue congelada) con lr / 10
        model.backbone.trainable = True
        if args.freeze_bn:
            congelar_bn(model)
        compilar(args.lr / 10)
        print(f"Fase 2: red completa, lr {args.lr / 10:g}", flush=True)
    t0 = time.time()
    h = model.fit(train_ds, epochs=1, verbose=0)
    f1t, acct, rect = medir(train_eval, y_tr)
    f1v, accv, recv = medir(val_ds, y_va)
    print(f"epoca {e}/{args.epochs} | perdida {h.history['loss'][0]:.4f} | "
          f"TRAIN F1 {f1t:.3f} acc {acct:.3f} rec.mala {rect:.3f} | "
          f"VAL F1 {f1v:.3f} acc {accv:.3f} rec.mala {recv:.3f} | "
          f"{time.time() - t0:.0f}s", flush=True)
