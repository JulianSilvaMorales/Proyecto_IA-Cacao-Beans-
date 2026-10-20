"""
Nombre: 13_probe_cv.py
Descripcion: Diagnostico. Validacion cruzada estratificada (5 particiones, semilla
             42) sobre train+val de una regresion logistica con las
             caracteristicas de EfficientNet-B0 congelado, comparada con un
             clasificador que usa solo la proporcion ancho/alto. NO usa test.
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
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")

SEED = 42
keras.utils.set_random_seed(SEED)
PRESET = "efficientnet_b0_ra4_e3600_r224_imagenet"
tmp = kh.models.EfficientNetImageClassifier.from_preset(PRESET, num_classes=2)
conv, backbone = tmp.preprocessor.image_converter, tmp.backbone

feats, dfs = [], []
for part in ["train", "val"]:
    df = pd.read_csv(f"data/splits/{part}.csv")
    fs = []
    for x, _ in pipe.make_dataset(part, training=False):   # sin augmentation, orden del CSV
        f = backbone(conv(x), training=False)
        if len(f.shape) == 4:
            f = tf.reduce_mean(f, axis=[1, 2])
        fs.append(f.numpy())
    feats.append(np.concatenate(fs))
    dfs.append(df)
X = np.concatenate(feats)
d = pd.concat(dfs, ignore_index=True)
inv = pd.read_csv("ml/results/inventory_crops.csv")[["path", "width", "height"]]
d = d.merge(inv, on="path", how="left")
y = d["label"].map(pipe.CLASS_TO_ID).values
ar = (d["width"] / d["height"]).values.reshape(-1, 1)
print(f"Imagenes: {len(d)} | mala: {int(y.sum())} | caracteristicas: {X.shape[1]}", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def cv(nombre, M, C):
    f1s, accs, recs = [], [], []
    for tr, te in skf.split(M, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(M[tr], y[tr])
        p = clf.predict(M[te])
        f1s.append(f1_score(y[te], p, average="macro"))
        accs.append(accuracy_score(y[te], p))
        recs.append(recall_score(y[te], p, pos_label=1))
    print(f"{nombre:28} | F1 {np.mean(f1s):.3f} +- {np.std(f1s):.3f} | "
          f"acc {np.mean(accs):.3f} | rec.mala {np.mean(recs):.3f}", flush=True)


cv("solo proporcion ancho/alto", ar, 1.0)
for C in [0.001, 0.01, 0.1]:
    cv(f"caracteristicas C={C}", X, C)
