"""
Nombre: 12_feature_probe.py
Descripcion: Diagnostico. Extrae caracteristicas de EfficientNet-B0 preentrenado
             (congelado, sin entrenar) para train y val sin augmentation, con 3
             preprocesamientos distintos, y ajusta una regresion logistica
             simple. Sirve para saber si las caracteristicas y el
             preprocesamiento llevan senal, sin depender del optimizador. NO usa test.
Fecha_creacion: 2026-10-19
Actualizacion: 2026-10-19
Autor: Julian Andres Silvas Morales
Version: 1.0
"""
import importlib
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import keras_hub as kh
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")

SEED = 42
keras.utils.set_random_seed(SEED)
PRESET = "efficientnet_b0_ra4_e3600_r224_imagenet"

tmp = kh.models.EfficientNetImageClassifier.from_preset(PRESET, num_classes=2)
conv = tmp.preprocessor.image_converter
backbone = tmp.backbone

MEAN = tf.constant([0.485, 0.456, 0.406])
STD = tf.constant([0.229, 0.224, 0.225])
VARIANTES = {
    "converter del preset": lambda x: conv(x),
    "ImageNet media/desv ": lambda x: (x / 255.0 - MEAN) / STD,
    "solo /255 (0 a 1)   ": lambda x: x / 255.0,
}


def caracteristicas(fn):
    out = {}
    for part in ["train", "val"]:
        fs, ys = [], []
        for x, y in pipe.make_dataset(part, training=False):
            f = backbone(fn(x), training=False)
            if len(f.shape) == 4:
                f = tf.reduce_mean(f, axis=[1, 2])
            fs.append(f.numpy())
            ys.append(y.numpy())
        out[part] = (np.concatenate(fs), np.concatenate(ys))
    return out


def metricas(y, p):
    return (f1_score(y, p, average="macro"), accuracy_score(y, p),
            recall_score(y, p, pos_label=1))


for nombre, fn in VARIANTES.items():
    d = caracteristicas(fn)
    (xt, yt), (xv, yv) = d["train"], d["val"]
    sc = StandardScaler().fit(xt)
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
    clf.fit(sc.transform(xt), yt)
    ft = metricas(yt, clf.predict(sc.transform(xt)))
    fv = metricas(yv, clf.predict(sc.transform(xv)))
    print(f"{nombre} | caract. {xt.shape[1]} | "
          f"TRAIN F1 {ft[0]:.3f} acc {ft[1]:.3f} rec.mala {ft[2]:.3f} | "
          f"VAL F1 {fv[0]:.3f} acc {fv[1]:.3f} rec.mala {fv[2]:.3f}", flush=True)
