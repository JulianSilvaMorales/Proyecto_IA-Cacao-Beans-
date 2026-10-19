"""
Nombre: 07_data_pipeline.py
Descripcion: Pipeline tf.data que lee data/splits/*.csv y entrega los recortes
             por lotes (data generator). Vuelve cuadrado cada recorte sin
             deformar el grano, prolongando el borde del recorte (el
             papel); augmentation SOLO en train. Verifica un lote y guarda una
             muestra visual en ml/results/pipeline_sample.png.
Fecha_creacion: 2026-10-19
Actualizacion: 2026-10-19
Autor: Julian Andres Silvas Morales
Version: 1.4
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = 224          # configurable: probar 384 mas adelante
BATCH_SIZE = 32
SPLITS = Path("data/splits")
OUT = Path("ml/results")
CLASS_TO_ID = {"sana": 0, "mala": 1}
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)


def _suavizar(fila, puntos=16):
    """Suaviza una fila de forma (1, n, 3): la reduce a 'puntos' valores
    (promedio por zonas) y la vuelve a ampliar. Diluye motas y rayas."""
    n = tf.shape(fila)[1]
    corta = tf.image.resize(fila, (1, puntos), method="area")
    return tf.image.resize(corta, (1, n), method="bilinear")


def load_image(path, label):
    """Lee el JPG y lo vuelve cuadrado prolongando hacia afuera el borde del
    recorte (promedio de 4 filas/columnas, suavizado), sin deformar el grano;
    luego lo escala a IMG_SIZE."""
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32)
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    lado = tf.maximum(h, w)
    pad_h, pad_w = lado - h, lado - w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    k = 4  # filas/columnas del borde que se promedian
    fila_sup = _suavizar(tf.reduce_mean(img[:k], axis=0, keepdims=True))
    fila_inf = _suavizar(tf.reduce_mean(img[-k:], axis=0, keepdims=True))
    img = tf.concat([tf.repeat(fila_sup, top, axis=0), img,
                     tf.repeat(fila_inf, bottom, axis=0)], axis=0)
    col_izq = tf.reduce_mean(img[:, :k], axis=1, keepdims=True)   # (alto, 1, 3)
    col_der = tf.reduce_mean(img[:, -k:], axis=1, keepdims=True)
    col_izq = tf.transpose(_suavizar(tf.transpose(col_izq, [1, 0, 2])), [1, 0, 2])
    col_der = tf.transpose(_suavizar(tf.transpose(col_der, [1, 0, 2])), [1, 0, 2])
    img = tf.concat([tf.repeat(col_izq, left, axis=1), img,
                     tf.repeat(col_der, right, axis=1)], axis=1)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return tf.clip_by_value(img, 0.0, 255.0), label


def augment(img, label):
    """Augmentation leve. Sin cambios de color: el color distingue sana de mala."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 15.0)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return tf.clip_by_value(img, 0.0, 255.0), label


def make_dataset(name, training=False):
    """Devuelve un tf.data.Dataset por lotes para la particion 'name'."""
    df = pd.read_csv(SPLITS / f"{name}.csv")
    missing = [p for p in df["path"] if not Path(p).exists()]
    assert not missing, f"{len(missing)} archivos no existen (ej. {missing[:2]})"
    paths = df["path"].values
    labels = df["label"].map(CLASS_TO_ID).values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, training in [("train", True), ("val", False), ("test", False)]:
        ds = make_dataset(name, training)
        n = sum(1 for _ in ds.unbatch())
        print(f"{name}: {n} imagenes, {len(ds)} lotes de hasta {BATCH_SIZE}")

    x, y = next(iter(make_dataset("train", True)))
    print("\nLote de train -> imagenes:", x.shape, x.dtype, "| etiquetas:", y.shape)
    print("Rango de pixeles: min %.1f, max %.1f" % (x.numpy().min(), x.numpy().max()))
    print("Etiquetas en el lote (0=sana, 1=mala):", y.numpy().tolist())

    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for ax, img, lab in zip(axes.ravel(), x.numpy(), y.numpy()):
        ax.imshow(img.astype("uint8"))
        ax.set_title("mala" if lab == 1 else "sana")
        ax.axis("off")
    fig.suptitle("Muestra de train (con augmentation)")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_sample.png", dpi=120)
    print("\nGuardado: ml/results/pipeline_sample.png")
