"""
Nombre: 10_train_federated.py
Descripcion: Entrenamiento con Federated Learning (FedAvg simulado) de un
             modelo preentrenado de KerasHub con 2 clases (sana/mala). Cada
             ronda: cada cliente entrena localmente sobre su parte de train,
             el servidor promedia los pesos (ponderado por tamano), y se mide
             F1 macro y exactitud en val. Guarda el mejor modelo por F1 macro en
             val y un checkpoint por ronda para reanudar. NO usa test.
Fecha_creacion: 2026-10-19
Actualizacion: 2026-10-19
Autor: Julian Andres Silvas Morales
Version: 1.4
Uso (desde la raiz del repo):
    python ml/src/10_train_federated.py --model EfficientNet-B0 --rounds 2 \\
        --local-epochs 1 --lr 1e-4 --tag smoke
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

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # make_dataset, CLASS_TO_ID

SEED = 42
N_CLIENTS = 5
RUNS = Path.home() / "cacao_runs"
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=MODELOS)
ap.add_argument("--rounds", type=int, default=10)
ap.add_argument("--local-epochs", type=int, default=1)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--tag", default="run")
ap.add_argument("--clipnorm", type=float, default=None)
ap.add_argument("--freeze-bn", action="store_true",
                help="congela BatchNorm (usa sus estadisticas de ImageNet)")
args = ap.parse_args()

keras.utils.set_random_seed(SEED)
run_dir = RUNS / f"{args.tag}_{args.model}"
run_dir.mkdir(parents=True, exist_ok=True)

# --- modelo y su preprocesamiento -------------------------------------------
cls, preset = MODELOS[args.model]
# El converter se toma de una instancia temporal; el modelo real se carga SIN
# preprocesador para que fit/predict no apliquen el preprocesamiento por segunda vez.
_tmp = cls.from_preset(preset, num_classes=2)
conv = _tmp.preprocessor.image_converter
del _tmp
model = cls.from_preset(preset, num_classes=2, activation="softmax", preprocessor=None)
print("preprocessor del modelo (debe ser None):", model.preprocessor)
print(f"Modelo: {args.model} ({preset}) | converter: {type(conv).__name__}")


def congelar_bn(layer):
    """Pone trainable=False a toda BatchNormalization (corre en modo inferencia)."""
    n = 1 if isinstance(layer, keras.layers.BatchNormalization) else 0
    if n:
        layer.trainable = False
    for sub in getattr(layer, "layers", None) or []:
        n += congelar_bn(sub)
    return n


if args.freeze_bn:
    print("BatchNorm congeladas:", congelar_bn(model))


def prep(ds, class_weights=None):
    """Aplica el preprocesamiento del modelo; opcionalmente agrega peso por muestra."""
    if class_weights is None:
        return ds.map(lambda x, y: (conv(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    w = tf.constant(class_weights, tf.float32)
    return ds.map(lambda x, y: (conv(x), y, tf.gather(w, y)),
                  num_parallel_calls=tf.data.AUTOTUNE)


# --- datos: 5 clientes (train repartido), val global -------------------------
client_ds, sizes = [], []
for k in range(N_CLIENTS):
    name = f"federated/client_{k}"
    df = pd.read_csv(f"data/splits/{name}.csv")
    n, n_mala = len(df), int((df["label"] == "mala").sum())
    cw = [n / (2 * (n - n_mala)), n / (2 * n_mala)]          # [sana, mala]
    client_ds.append(prep(pipe.make_dataset(name, training=True), cw))
    sizes.append(n)
print("Tamano de cada cliente:", sizes, "| peso de clase [sana, mala]:",
      [round(v, 2) for v in cw])

val_df = pd.read_csv("data/splits/val.csv")
y_val = val_df["label"].map(pipe.CLASS_TO_ID).values
val_ds = prep(pipe.make_dataset("val", training=False))

# --- comprobaciones iniciales (verifican los supuestos) ----------------------
xb, yb, wb = next(iter(client_ds[0]))
pb = model(xb, training=False).numpy()
print("Lote:", xb.shape, "| salida:", pb.shape,
      "| suma de probabilidades por fila (debe ser ~1):", round(float(pb.sum(1).mean()), 3))


def evaluar():
    pred = model.predict(val_ds, verbose=0).argmax(1)
    return {"f1_macro": f1_score(y_val, pred, average="macro"),
            "acc": accuracy_score(y_val, pred),
            "recall_mala": recall_score(y_val, pred, pos_label=1)}


# --- reanudar si hay checkpoint ---------------------------------------------
state_f, hist_f = run_dir / "state.json", run_dir / "history.csv"
state = {"round": 0, "best_f1": -1.0, "best_round": 0}
history = []
if state_f.exists():
    state = json.loads(state_f.read_text())
    history = pd.read_csv(hist_f).to_dict("records")
    model.load_weights(run_dir / "last.weights.h5")
    print(f"Reanudando desde la ronda {state['round']} (mejor F1 val {state['best_f1']:.4f})")

# Se compila UNA sola vez: recompilar por cliente costaba ~20 s extra por cliente
model.compile(optimizer=keras.optimizers.Adam(args.lr, clipnorm=args.clipnorm),
              loss=keras.losses.SparseCategoricalCrossentropy(), jit_compile=False)

def reiniciar_optimizador(m):
    """Pone a cero los momentos de Adam y el contador de pasos SIN tocar la tasa
    de aprendizaje (que tambien es una variable del optimizador)."""
    for v in m.optimizer.variables:
        nombre = (getattr(v, "path", None) or v.name).lower()
        if "learning_rate" in nombre:
            continue
        v.assign(tf.zeros_like(v))
    lr_actual = float(m.optimizer.learning_rate)
    assert abs(lr_actual - args.lr) < 1e-12, f"la tasa de aprendizaje quedo en {lr_actual}"


# --- FedAvg ----------------------------------------------------------------
for r in range(state["round"] + 1, args.rounds + 1):
    t0 = time.time()
    g = model.get_weights()
    locales = []
    for k in range(N_CLIENTS):
        model.set_weights(g)
        reiniciar_optimizador(model)             # optimizador "nuevo" por cliente
        model.fit(client_ds[k], epochs=args.local_epochs, verbose=0)
        locales.append(model.get_weights())
    promedio = [np.average(np.stack(ws), axis=0, weights=sizes).astype(ws[0].dtype)
                for ws in zip(*locales)]
    model.set_weights(promedio)

    m = evaluar()
    history.append({"ronda": r, **m, "seg": round(time.time() - t0, 1)})
    model.save_weights(run_dir / "last.weights.h5")
    if m["f1_macro"] > state["best_f1"]:
        state["best_f1"], state["best_round"] = m["f1_macro"], r
        model.save_weights(run_dir / "best.weights.h5")
    state["round"] = r
    state_f.write_text(json.dumps(state))
    pd.DataFrame(history).to_csv(hist_f, index=False)
    print(f"ronda {r:>3}/{args.rounds} | val F1 macro {m['f1_macro']:.4f} | "
          f"acc {m['acc']:.4f} | recall mala {m['recall_mala']:.4f} | "
          f"{time.time() - t0:.0f}s", flush=True)

print(f"\nFIN. Mejor ronda {state['best_round']} con F1 macro en val {state['best_f1']:.4f}")
print(f"Resultados en {run_dir}")
