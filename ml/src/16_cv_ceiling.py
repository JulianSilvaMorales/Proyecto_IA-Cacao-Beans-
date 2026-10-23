"""
Nombre: 16_cv_ceiling.py
Descripcion: Mide el techo de rendimiento con validacion cruzada estratificada de
             5 particiones sobre train+val (1300 recortes), entrenamiento
             CENTRALIZADO con la misma configuracion de la cuadricula (BatchNorm
             congeladas, clipnorm 1.0, logits). Guarda las probabilidades fuera de
             particion (OOF) de cada epoca y calcula F1 macro, exactitud y recall
             por clase con intervalos de confianza por bootstrap. Opciones para
             comparar el tamano de entrada (--img-size) y el sobremuestreo de la
             clase mala con augmentation (--oversample). NO usa test y NO cambia
             data/splits/ (las particiones se generan en memoria, semilla 42).
Fecha_creacion: 2026-10-23
Actualizacion: 2026-10-23
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo):
    python -u ml/src/16_cv_ceiling.py --model EfficientNet-B0 --lr 1e-5 --epochs 15
    python -u ml/src/16_cv_ceiling.py --model EfficientNet-B0 --lr 1e-5 --img-size 384
    python -u ml/src/16_cv_ceiling.py --model EfficientNet-B0 --lr 1e-5 --oversample
"""
import argparse
import importlib
import json
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
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # load_image, augment, CLASS_TO_ID

SEED = 42
N_FOLDS = 5
N_BOOT = 2000
AUTO = tf.data.AUTOTUNE
OUT = Path("ml/results/cv_ceiling")          # resumenes livianos (versionados)
RUNS = Path.home() / "cacao_runs" / "cv"     # probabilidades OOF (fuera del repo)
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=MODELOS)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--img-size", type=int, default=224)
ap.add_argument("--oversample", action="store_true",
                help="repite la clase mala (con augmentation) hasta igualar a sana, sin pesos de clase")
args = ap.parse_args()
if args.img_size != 224 and args.model in ("Swin-Tiny", "DeiT-Small"):
    sys.exit("Swin-Tiny y DeiT-Small solo se prueban a 224 (embeddings de posicion fijos)")

pipe.IMG_SIZE = args.img_size     # load_image lee esta global al construir el grafo
tag = f"{args.model}_{args.img_size}{'_os' if args.oversample else ''}"
OUT.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
print(f"== {tag} | lr {args.lr:g} | {args.epochs} epocas | {N_FOLDS} particiones", flush=True)

# --- datos: train + val (1300); test NO se lee --------------------------------
df = pd.concat([pd.read_csv("data/splits/train.csv"), pd.read_csv("data/splits/val.csv")],
               ignore_index=True)
assert len(df) == 1300 and df["path"].is_unique, "se esperaban 1300 recortes unicos"
y = df["label"].map(pipe.CLASS_TO_ID).values.astype("int32")
print(f"recortes: {len(df)} | mala: {int(y.sum())} | sana: {int((y == 0).sum())}")

# --- preprocesamiento del preset, sin el reescalado del converter -------------
cls, preset = MODELOS[args.model]
_tmp = cls.from_preset(preset, num_classes=2)
conv = _tmp.preprocessor.image_converter
del _tmp
scale = tf.constant(np.asarray(conv.scale, np.float32))
offset = tf.constant(np.asarray(conv.offset, np.float32))


def preprocesar(x):
    """Igual que el image_converter del preset pero sin reescalar a 224."""
    return x * scale + offset


_x = tf.random.uniform((2, 224, 224, 3), 0, 255)
assert np.allclose(preprocesar(_x).numpy(), conv(_x).numpy(), atol=1e-4), \
    "el preprocesamiento manual no coincide con el converter del preset"
print("preprocesamiento manual == converter del preset (comprobado a 224)")


def make_ds(part, training, weights=None):
    paths, labels = part["path"].values, part["label"].map(pipe.CLASS_TO_ID).values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(part), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(pipe.load_image, num_parallel_calls=AUTO)
    if training:
        ds = ds.map(pipe.augment, num_parallel_calls=AUTO)
    if weights is None:
        ds = ds.map(lambda a, b: (preprocesar(a), b), num_parallel_calls=AUTO)
    else:
        w = tf.constant(weights, tf.float32)
        ds = ds.map(lambda a, b: (preprocesar(a), b, tf.gather(w, b)), num_parallel_calls=AUTO)
    return ds.batch(pipe.BATCH_SIZE).prefetch(AUTO)


def congelar_bn(m):
    """Congela TODAS las BatchNormalization (igual que 10 v1.6 / 11 v1.5)."""
    bns = [l for l in m._flatten_layers(include_self=False, recursive=True)
           if isinstance(l, keras.layers.BatchNormalization)]
    for l in bns:
        l.trainable = False
    return len(bns)


def metricas(yt, p_mala, umbral=0.5):
    pred = (p_mala >= umbral).astype(int)
    return {"f1_macro": f1_score(yt, pred, average="macro"),
            "acc": accuracy_score(yt, pred),
            "recall_mala": recall_score(yt, pred, pos_label=1),
            "recall_sana": recall_score(yt, pred, pos_label=0)}


# --- validacion cruzada ---------------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.full((args.epochs, len(df)), np.nan, np.float32)    # P(mala) por epoca
filas = []
t_total = time.time()
for k, (i_tr, i_te) in enumerate(skf.split(df, y)):
    keras.backend.clear_session()
    keras.utils.set_random_seed(SEED + k)
    tr, te = df.iloc[i_tr], df.iloc[i_te]
    n, n_mala = len(tr), int((tr["label"] == "mala").sum())
    if args.oversample:
        rep = int(round((n - n_mala) / n_mala))
        tr = pd.concat([tr[tr["label"] == "sana"]] + [tr[tr["label"] == "mala"]] * rep)
        train_ds = make_ds(tr, training=True)
        info = f"sobremuestreo mala x{rep} -> {len(tr)} imagenes, sin pesos de clase"
    else:
        cw = [n / (2 * (n - n_mala)), n / (2 * n_mala)]
        train_ds = make_ds(tr, training=True, weights=cw)
        info = f"pesos de clase [sana, mala] = {[round(v, 2) for v in cw]}"
    test_ds = make_ds(te, training=False)

    model = cls.from_preset(preset, num_classes=2, activation=None, preprocessor=None)
    n_bn = congelar_bn(model)
    model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=1.0),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    print(f"\n-- particion {k + 1}/{N_FOLDS}: train {len(i_tr)} | eval {len(i_te)} "
          f"({int(y[i_te].sum())} mala) | BN congeladas {n_bn} | {info}", flush=True)
    for e in range(args.epochs):
        t0 = time.time()
        h = model.fit(train_ds, epochs=1, verbose=0)
        p = tf.nn.softmax(model.predict(test_ds, verbose=0)).numpy()[:, 1]
        oof[e, i_te] = p
        m = metricas(y[i_te], p)
        filas.append({"particion": k + 1, "epoca": e + 1, "perdida": h.history["loss"][0], **m})
        print(f"epoca {e + 1:>2}/{args.epochs} | perdida {h.history['loss'][0]:.4f} | "
              f"F1 {m['f1_macro']:.3f} acc {m['acc']:.3f} rec.mala {m['recall_mala']:.3f} "
              f"rec.sana {m['recall_sana']:.3f} | {time.time() - t0:.0f}s", flush=True)
    del model

assert not np.isnan(oof).any(), "quedaron recortes sin prediccion fuera de particion"
np.savez_compressed(RUNS / f"{tag}_oof.npz", oof=oof, y=y, path=df["path"].values)
hist = pd.DataFrame(filas)
hist.to_csv(OUT / f"{tag}_folds.csv", index=False)

# --- resumen: epoca final y mejor epoca segun la media de las particiones ------
curva = hist.groupby("epoca")["f1_macro"].agg(["mean", "std"])
e_mejor = int(curva["mean"].idxmax())
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))


def resumen(e, etiqueta):
    p = oof[e - 1]
    m = metricas(y, p)
    por_part = hist[hist["epoca"] == e]
    lineas = [f"{etiqueta} (epoca {e}):"]
    boot = pd.DataFrame([metricas(y[ix], p[ix]) for ix in boot_idx])   # mismas remuestras
    for clave in ("f1_macro", "acc", "recall_mala", "recall_sana"):
        lo, hi = np.percentile(boot[clave], [2.5, 97.5])
        lineas.append(f"  {clave:<12} OOF {m[clave]:.3f}  IC95 [{lo:.3f}, {hi:.3f}]  | "
                      f"por particion {por_part[clave].mean():.3f} +- {por_part[clave].std():.3f}")
    umbrales = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (p >= u).astype(int), average="macro") for u in umbrales]
    u = umbrales[int(np.argmax(f1s))]
    mu = metricas(y, p, u)
    lineas.append(f"  umbral que maximiza F1 (OPTIMISTA, elegido con los mismos datos): {u:.2f} -> "
                  f"F1 {mu['f1_macro']:.3f} acc {mu['acc']:.3f} rec.mala {mu['recall_mala']:.3f} "
                  f"rec.sana {mu['recall_sana']:.3f}")
    return lineas, m


txt = [f"Nombre: {tag}_resumen.txt",
       f"Descripcion: validacion cruzada de 5 particiones sobre train+val (1300), {tag}, "
       f"lr {args.lr:g}, {args.epochs} epocas. Generado por 16_cv_ceiling.py v1.0. NO usa test.",
       f"Fecha: {time.strftime('%Y-%m-%d')} | tiempo total {(time.time() - t_total) / 60:.1f} min",
       "",
       "F1 macro medio por epoca (media +- desv de las 5 particiones):",
       "  " + " ".join(f"{i}:{r['mean']:.3f}" for i, r in curva.iterrows()),
       ""]
l_fin, m_fin = resumen(args.epochs, "EPOCA FINAL")
l_mej, _ = resumen(e_mejor, "MEJOR EPOCA SEGUN LA MEDIA (algo optimista)")
txt += l_fin + [""] + l_mej
(OUT / f"{tag}_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")
(OUT / f"{tag}_resumen.json").write_text(json.dumps(
    {"tag": tag, "lr": args.lr, "epochs": args.epochs, "mejor_epoca": e_mejor,
     "final": {k: round(float(v), 4) for k, v in m_fin.items()}}, indent=2), encoding="utf8")
print("\n" + "\n".join(txt))
print(f"\nGuardado: {OUT}/{tag}_folds.csv, _resumen.txt, _resumen.json | OOF en {RUNS}")
