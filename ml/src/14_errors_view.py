"""
Nombre: 14_errors_view.py
Descripcion: Diagnostico. Con el clasificador lineal sobre caracteristicas de
             EfficientNet-B0 congelado (entrenado en train), guarda en
             ml/results/errors_val.png una cuadricula de val: errores y
             aciertos de cada clase, para revisarlos a ojo. NO usa test.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")
keras.utils.set_random_seed(42)

tmp = kh.models.EfficientNetImageClassifier.from_preset(
    "efficientnet_b0_ra4_e3600_r224_imagenet", num_classes=2)
conv, backbone = tmp.preprocessor.image_converter, tmp.backbone


def feats(part):
    fs = []
    for x, _ in pipe.make_dataset(part, training=False):
        f = backbone(conv(x), training=False)
        if len(f.shape) == 4:
            f = tf.reduce_mean(f, axis=[1, 2])
        fs.append(f.numpy())
    return np.concatenate(fs)


tr, va = pd.read_csv("data/splits/train.csv"), pd.read_csv("data/splits/val.csv")
ytr = tr["label"].map(pipe.CLASS_TO_ID).values
yva = va["label"].map(pipe.CLASS_TO_ID).values
clf = make_pipeline(StandardScaler(), LogisticRegression(
    C=0.001, max_iter=3000, class_weight="balanced"))
clf.fit(feats("train"), ytr)
pred = clf.predict(feats("val"))

rng = np.random.default_rng(42)
grupos = [
    ("mala bien clasificada", (yva == 1) & (pred == 1)),
    ("mala clasificada como sana", (yva == 1) & (pred == 0)),
    ("sana bien clasificada", (yva == 0) & (pred == 0)),
    ("sana clasificada como mala", (yva == 0) & (pred == 1)),
]
fig, axes = plt.subplots(4, 5, figsize=(15, 11))
for fila, (titulo, mask) in enumerate(grupos):
    idx = np.where(mask)[0]
    sel = rng.choice(idx, size=min(5, len(idx)), replace=False)
    for col in range(5):
        ax = axes[fila, col]
        ax.axis("off")
        if col < len(sel):
            ax.imshow(Image.open(va.iloc[sel[col]]["path"]).convert("RGB"))
        if col == 0:
            ax.set_title(f"{titulo} (n={int(mask.sum())})", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig("ml/results/errors_val.png", dpi=90)
print("Guardado ml/results/errors_val.png")
