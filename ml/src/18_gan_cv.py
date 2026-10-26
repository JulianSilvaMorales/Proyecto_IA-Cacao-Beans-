"""
Nombre: 18_gan_cv.py
Descripcion: Experimento de GAN (Topic 1) para la clase minoritaria. En cada una de
             las 5 particiones de la validacion cruzada de 16_cv_ceiling.py (mismas
             particiones, semilla 42) entrena una DCGAN 128x128 con spectral
             normalization y DiffAugment SOLO con las mala del train de esa
             particion, genera mala sinteticas hasta igualar a sana y entrena
             EfficientNet-B0 (misma configuracion que la CV) con reales + sinteticas.
             Compara de forma pareada con la linea base de la CV (sin GAN) mediante
             bootstrap de la diferencia de F1 macro. NO usa test y NO cambia
             data/splits/.
Fecha_creacion: 2026-10-26
Actualizacion: 2026-10-26
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo; requiere ~/cacao_runs/cv/EfficientNet-B0_224_oof.npz):
    python -u ml/src/18_gan_cv.py
    python -u ml/src/18_gan_cv.py --folds 1 --gan-steps 50 --epochs 1   # prueba rapida
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # load_image, augment, CLASS_TO_ID

SEED = 42
N_FOLDS = 5
N_BOOT = 2000
AUTO = tf.data.AUTOTUNE
GAN_SIZE = 128
Z_DIM = 128
PRESET = "efficientnet_b0_ra4_e3600_r224_imagenet"
OUT = Path("ml/results/gan")                          # resumenes y muestras (versionados)
RUNS = Path.home() / "cacao_runs" / "gan"             # OOF y generadores (fuera del repo)
BASE_OOF = Path.home() / "cacao_runs" / "cv" / "EfficientNet-B0_224_oof.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--gan-steps", type=int, default=6000)
ap.add_argument("--gan-batch", type=int, default=32)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--folds", type=int, default=N_FOLDS, help="particiones a ejecutar (prueba rapida)")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

# --- datos: train + val (1300), mismo orden que 16_cv_ceiling.py; test NO se lee
df = pd.concat([pd.read_csv("data/splits/train.csv"), pd.read_csv("data/splits/val.csv")],
               ignore_index=True)
y = df["label"].map(pipe.CLASS_TO_ID).values.astype("int32")
base = np.load(BASE_OOF, allow_pickle=True)
assert (base["path"] == df["path"].values).all(), "el orden no coincide con la CV base"
p_base = base["oof"][-1]                                   # epoca final de la linea base
print(f"recortes: {len(df)} | linea base: {BASE_OOF.name} (epoca {base['oof'].shape[0]})")

_tmp = kh.models.EfficientNetImageClassifier.from_preset(PRESET, num_classes=2)
conv = _tmp.preprocessor.image_converter
del _tmp


def cargar(paths):
    """Recortes con el relleno del pipeline (224, 0-255), como array float32."""
    ds = tf.data.Dataset.from_tensor_slices((paths, np.zeros(len(paths), "int32")))
    ds = ds.map(pipe.load_image, num_parallel_calls=AUTO).map(lambda a, b: a).batch(64)
    return np.concatenate([b.numpy() for b in ds])


# --- GAN: DCGAN 128x128, spectral normalization en D, DiffAugment ---------------
def generador():
    z = keras.Input((Z_DIM,))
    x = layers.Dense(8 * 8 * 512, use_bias=False)(z)
    x = layers.ReLU()(layers.BatchNormalization()(x))
    x = layers.Reshape((8, 8, 512))(x)
    for f in (256, 128, 64, 32):                                  # 16, 32, 64, 128
        x = layers.Conv2DTranspose(f, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.ReLU()(layers.BatchNormalization()(x))
    return keras.Model(z, layers.Conv2D(3, 3, padding="same", activation="tanh")(x), name="G")


def discriminador():
    i = keras.Input((GAN_SIZE, GAN_SIZE, 3))
    x = i
    for f in (32, 64, 128, 256, 512):                             # 64, 32, 16, 8, 4
        x = layers.SpectralNormalization(layers.Conv2D(f, 4, strides=2, padding="same"))(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.SpectralNormalization(layers.Dense(1))(layers.Flatten()(x))
    return keras.Model(i, x, name="D")


def diffaug(x):
    """DiffAugment (Zhao et al., 2020): color, traslacion 1/8 y cutout 1/2, por muestra.
    Se aplica a imagenes reales y falsas antes de D; x en [-1, 1]."""
    n = tf.shape(x)[0]
    x = x + tf.random.uniform([n, 1, 1, 1], -0.5, 0.5)                       # brillo
    m = tf.reduce_mean(x, axis=3, keepdims=True)
    x = (x - m) * tf.random.uniform([n, 1, 1, 1], 0.0, 2.0) + m             # saturacion
    m = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    x = (x - m) * tf.random.uniform([n, 1, 1, 1], 0.5, 1.5) + m             # contraste
    s = GAN_SIZE // 8
    xp = tf.pad(x, [[0, 0], [s, s], [s, s], [0, 0]], mode="REFLECT")
    idx = tf.range(GAN_SIZE)
    oy = tf.random.uniform([n], 0, 2 * s + 1, tf.int32)
    ox = tf.random.uniform([n], 0, 2 * s + 1, tf.int32)
    x = tf.gather(xp, oy[:, None] + idx[None, :], axis=1, batch_dims=1)
    x = tf.gather(x, ox[:, None] + idx[None, :], axis=2, batch_dims=1)
    c = GAN_SIZE // 4                                                         # mitad del lado
    cy = tf.random.uniform([n, 1, 1], 0, GAN_SIZE, tf.int32)
    cx = tf.random.uniform([n, 1, 1], 0, GAN_SIZE, tf.int32)
    fuera = (tf.abs(idx[None, :, None] - cy) >= c) | (tf.abs(idx[None, None, :] - cx) >= c)
    return x * tf.cast(fuera, x.dtype)[..., None]


def entrenar_gan(reales_128, k):
    G, D = generador(), discriminador()
    g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
    d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
    g_opt.build(G.trainable_variables)
    d_opt.build(D.trainable_variables)

    @tf.function
    def paso(real):
        z = tf.random.normal([tf.shape(real)[0], Z_DIM])
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = G(z, training=True)
            l_real = D(diffaug(real), training=True)
            l_fake = D(diffaug(fake), training=True)
            d_loss = tf.reduce_mean(tf.nn.softplus(-l_real)) + tf.reduce_mean(tf.nn.softplus(l_fake))
            g_loss = tf.reduce_mean(tf.nn.softplus(-l_fake))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, D.trainable_variables), D.trainable_variables))
        g_opt.apply_gradients(zip(gt.gradient(g_loss, G.trainable_variables), G.trainable_variables))
        return d_loss, g_loss

    ds = (tf.data.Dataset.from_tensor_slices(reales_128)
          .shuffle(len(reales_128), seed=SEED + k).repeat()
          .map(lambda a: tf.image.random_flip_up_down(tf.image.random_flip_left_right(a)))
          .batch(args.gan_batch, drop_remainder=True).prefetch(AUTO))
    t0 = time.time()
    for i, real in enumerate(ds.take(args.gan_steps), start=1):
        dl, gl = paso(real)
        if i % 1000 == 0 or i == args.gan_steps:
            print(f"   GAN paso {i}/{args.gan_steps} | perdida D {float(dl):.3f} G {float(gl):.3f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
    G.save_weights(RUNS / f"G_particion{k + 1}.weights.h5")
    return G


def sinteticas(G, n):
    """n mala sinteticas en 224x224, rango 0-255 (como el pipeline)."""
    out = []
    for i in range(0, n, 64):
        z = tf.random.normal([min(64, n - i), Z_DIM])
        x = (G(z, training=False) + 1.0) * 127.5
        out.append(tf.image.resize(x, (224, 224)).numpy())
    return np.clip(np.concatenate(out), 0, 255).astype("float32")


def muestras(fake, real, k):
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
    for j, ax in enumerate(axes.ravel()):
        img = fake[j] if j < 16 else real[j - 16]
        ax.imshow(img.astype("uint8"))
        ax.set_title("sintetica" if j < 16 else "real", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Particion {k + 1}: mala sinteticas (filas 1-2) y reales (fila 3)")
    fig.tight_layout()
    fig.savefig(OUT / f"muestras_particion{k + 1}.png", dpi=80)
    plt.close(fig)


# --- clasificador (misma configuracion que 16_cv_ceiling.py) -----------------------
def congelar_bn(m):
    bns = [l for l in m._flatten_layers(include_self=False, recursive=True)
           if isinstance(l, keras.layers.BatchNormalization)]
    for l in bns:
        l.trainable = False
    return len(bns)


def metricas(yt, p, umbral=0.5):
    pred = (p >= umbral).astype(int)
    return {"f1_macro": f1_score(yt, pred, average="macro"), "acc": accuracy_score(yt, pred),
            "recall_mala": recall_score(yt, pred, pos_label=1),
            "recall_sana": recall_score(yt, pred, pos_label=0)}


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.full(len(df), np.nan, np.float32)
filas = []
t_total = time.time()
for k, (i_tr, i_te) in enumerate(skf.split(df, y)):
    if k >= args.folds:
        break
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + k)
    tr, te = df.iloc[i_tr], df.iloc[i_te]
    mala_tr = tr[tr["label"] == "mala"]["path"].values
    n_sana, n_mala = int((tr["label"] == "sana").sum()), len(mala_tr)
    print(f"\n-- particion {k + 1}/{N_FOLDS}: train {len(tr)} ({n_mala} mala) | eval {len(te)}", flush=True)

    reales = cargar(mala_tr)                                         # 224, 0-255
    reales_128 = tf.image.resize(reales, (GAN_SIZE, GAN_SIZE)).numpy() / 127.5 - 1.0
    G = entrenar_gan(reales_128.astype("float32"), k)
    n_syn = n_sana - n_mala
    syn = sinteticas(G, n_syn)
    muestras(syn, reales, k)
    del G
    print(f"   {n_syn} mala sinteticas generadas -> train {len(tr) + n_syn} "
          f"({n_sana} sana / {n_mala} + {n_syn} mala)", flush=True)

    keras.backend.clear_session()
    n_tot, n_mala_tot = len(tr) + n_syn, n_mala + n_syn
    cw = tf.constant([n_tot / (2 * (n_tot - n_mala_tot)), n_tot / (2 * n_mala_tot)], tf.float32)
    real_ds = tf.data.Dataset.from_tensor_slices(
        (tr["path"].values, tr["label"].map(pipe.CLASS_TO_ID).values.astype("int32"))
    ).map(pipe.load_image, num_parallel_calls=AUTO)
    syn_ds = tf.data.Dataset.from_tensor_slices((syn, np.ones(n_syn, "int32")))
    train_ds = (real_ds.concatenate(syn_ds).shuffle(n_tot, seed=SEED + k, reshuffle_each_iteration=True)
                .map(pipe.augment, num_parallel_calls=AUTO)
                .map(lambda a, b: (conv(a), b, tf.gather(cw, b)), num_parallel_calls=AUTO)
                .batch(pipe.BATCH_SIZE).prefetch(AUTO))
    test_ds = (tf.data.Dataset.from_tensor_slices(
        (te["path"].values, te["label"].map(pipe.CLASS_TO_ID).values.astype("int32")))
        .map(pipe.load_image, num_parallel_calls=AUTO)
        .map(lambda a, b: (conv(a), b), num_parallel_calls=AUTO)
        .batch(pipe.BATCH_SIZE).prefetch(AUTO))

    model = kh.models.EfficientNetImageClassifier.from_preset(
        PRESET, num_classes=2, activation=None, preprocessor=None)
    congelar_bn(model)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    for e in range(args.epochs):
        t0 = time.time()
        h = model.fit(train_ds, epochs=1, verbose=0)
        p = tf.nn.softmax(model.predict(test_ds, verbose=0)).numpy()[:, 1]
        m = metricas(y[i_te], p)
        filas.append({"particion": k + 1, "epoca": e + 1, "perdida": h.history["loss"][0], **m})
        print(f"   epoca {e + 1:>2}/{args.epochs} | perdida {h.history['loss'][0]:.4f} | "
              f"F1 {m['f1_macro']:.3f} rec.mala {m['recall_mala']:.3f} "
              f"rec.sana {m['recall_sana']:.3f} | {time.time() - t0:.0f}s", flush=True)
    oof[i_te] = p
    del model, syn

# --- comparacion pareada con la linea base (mismos granos, mismas particiones) ------
hechos = ~np.isnan(oof)
yh, pg, pb = y[hechos], oof[hechos], p_base[hechos]
np.savez_compressed(RUNS / "gan_oof.npz", oof=oof, y=y, path=df["path"].values)
pd.DataFrame(filas).to_csv(OUT / "gan_folds.csv", index=False)
rng = np.random.default_rng(SEED)
boot = rng.integers(0, len(yh), size=(N_BOOT, len(yh)))
mg, mb = metricas(yh, pg), metricas(yh, pb)
txt = [
    "Nombre: gan_resumen.txt",
    f"Descripcion: experimento de GAN (DCGAN 128 + DiffAugment, entrenada solo con las mala del "
    f"train de cada particion) frente a la linea base sin GAN. EfficientNet-B0, 224, lr {args.lr:g}, "
    f"{args.epochs} epocas, {args.gan_steps} pasos de GAN. Generado por 18_gan_cv.py v1.0. NO usa test.",
    f"Fecha: {time.strftime('%Y-%m-%d')} | particiones {int(args.folds)} | granos evaluados "
    f"{int(hechos.sum())} | tiempo {(time.time() - t_total) / 60:.0f} min",
    "",
    f"{'metrica':<12} {'con GAN':>10} {'sin GAN':>10} {'diferencia':>11}  IC95 de la diferencia (bootstrap pareado)",
]
dif = pd.DataFrame([pd.Series(metricas(yh[ix], pg[ix])) - pd.Series(metricas(yh[ix], pb[ix]))
                    for ix in boot])                      # mismas remuestras para ambos
for clave in ("f1_macro", "acc", "recall_mala", "recall_sana"):
    lo, hi = np.percentile(dif[clave], [2.5, 97.5])
    txt.append(f"{clave:<12} {mg[clave]:>10.3f} {mb[clave]:>10.3f} {mg[clave] - mb[clave]:>+11.3f}  "
               f"[{lo:+.3f}, {hi:+.3f}]")
txt += ["", "Lectura: la GAN ayuda solo si el IC95 de la diferencia de F1 macro queda por encima de 0.",
        "Muestras visuales: ml/results/gan/muestras_particion<k>.png"]
(OUT / "gan_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")
print("\n" + "\n".join(txt))
