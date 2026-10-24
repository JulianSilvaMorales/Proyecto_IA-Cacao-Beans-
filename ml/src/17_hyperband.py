"""
Nombre: 17_hyperband.py
Descripcion: Ajuste de hiperparametros con Hyperband (keras-tuner) para uno de
             los 5 modelos, entrenamiento CENTRALIZADO sobre train (1105) y
             objetivo F1 macro en val (195). Busca la tasa de aprendizaje
             (continua, escala log), el weight decay de AdamW y el numero de
             epocas (lo reparte Hyperband). Fija lo ya decidido con evidencia:
             todas las BatchNorm congeladas, clipnorm 1.0, logits, 224 px, pesos
             de clase. Reanuda si se corta (mismo comando). NO usa test.
             v1.1: se omite la configuracion de TensorBoard de keras-tuner (importaba
             tensorboard, que no esta instalado, aunque no se use).
             v1.2: libera la memoria de GPU entre pruebas (clear_session + gc.collect)
             e imprime la memoria en uso; ResNet-50 se quedaba sin memoria (OOM)
             tras 14 pruebas.
             v1.3: la memoria seguia creciendo ~310 MiB por prueba en ResNet-50
             (keras-tuner retiene cada modelo). El proceso termina con codigo 3
             cada --trials-per-process pruebas (el sistema libera la GPU) y se
             relanza con el mismo comando hasta que la busqueda acaba (codigo 0).
             v1.4: opcion --jit (compilacion "auto", con XLA, como en 16_cv_ceiling.py).
             Sin XLA, Swin-Tiny con lote 32 no cabe en los 8 GB de la GPU (OOM en la
             primera epoca); con XLA si cabe. Los demas modelos se ajustaron sin XLA.
Fecha_creacion: 2026-10-23
Actualizacion: 2026-10-24
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.4
Uso (desde la raiz del repo), repitiendo mientras el codigo de salida sea 3:
    until python -u ml/src/17_hyperband.py --model EfficientNet-B0; [ $? -ne 3 ]; do :; done
"""
import argparse
import gc
import importlib
import os
import sys
import time
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import keras_hub as kh
import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

# keras-tuner 1.4.8 importa tensorboard en cada prueba solo para configurar el callback
# TensorBoard; no lo usamos y tensorboard no esta instalado, asi que se omite ese paso.
kt.Tuner._configure_tensorboard_dir = lambda self, callbacks, trial, execution=0: None

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # make_dataset, CLASS_TO_ID

SEED = 42
AUTO = tf.data.AUTOTUNE
OUT = Path("ml/results/hyperband")                   # resumenes livianos (versionados)
RUNS = Path.home() / "cacao_runs" / "hyperband"      # estado del tuner y pesos (fuera del repo)
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=MODELOS)
ap.add_argument("--max-epochs", type=int, default=20)
ap.add_argument("--factor", type=int, default=3)
ap.add_argument("--trials-per-process", type=int, default=5,
                help="pruebas por proceso antes de salir con codigo 3 para liberar la GPU")
ap.add_argument("--jit", action="store_true",
                help='compila con jit_compile="auto" (XLA): menos memoria, mas tiempo de compilacion')
args = ap.parse_args()
keras.utils.set_random_seed(SEED)
OUT.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

# --- datos (igual que 11_central_check.py) -------------------------------------
cls, preset = MODELOS[args.model]
_tmp = cls.from_preset(preset, num_classes=2)
conv = _tmp.preprocessor.image_converter
del _tmp
tr_df = pd.read_csv("data/splits/train.csv")
va_df = pd.read_csv("data/splits/val.csv")
n, n_mala = len(tr_df), int((tr_df["label"] == "mala").sum())
cw = tf.constant([n / (2 * (n - n_mala)), n / (2 * n_mala)], tf.float32)
y_va = va_df["label"].map(pipe.CLASS_TO_ID).values
train_ds = pipe.make_dataset("train", training=True).map(
    lambda x, y: (conv(x), y, tf.gather(cw, y)), num_parallel_calls=AUTO)
val_ds = pipe.make_dataset("val", training=False).map(
    lambda x, y: (conv(x), y), num_parallel_calls=AUTO).cache()


def congelar_bn(m):
    """Congela TODAS las BatchNormalization (igual que 10 v1.6 / 11 v1.5)."""
    bns = [l for l in m._flatten_layers(include_self=False, recursive=True)
           if isinstance(l, keras.layers.BatchNormalization)]
    for l in bns:
        l.trainable = False
    return len(bns)


class F1Val(keras.callbacks.Callback):
    """Calcula el F1 macro en val al final de cada epoca y lo agrega a los logs."""

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(val_ds, verbose=0).argmax(1)
        logs["val_f1_macro"] = f1_score(y_va, pred, average="macro")


class Modelo(kt.HyperModel):
    def build(self, hp):
        # Los modelos de Keras tienen referencias circulares: sin gc.collect() la memoria
        # de GPU de las pruebas anteriores no se libera aunque se llame a clear_session.
        keras.backend.clear_session()
        gc.collect()
        mem = tf.config.experimental.get_memory_info("GPU:0")["current"] / 2**20
        print(f"[memoria GPU en uso al construir la prueba: {mem:.0f} MiB]", flush=True)
        lr = hp.Float("lr", 5e-6, 1e-4, sampling="log")
        wd = hp.Choice("weight_decay", [0.0, 1e-4, 1e-2])
        model = cls.from_preset(preset, num_classes=2, activation=None, preprocessor=None)
        congelar_bn(model)
        model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=wd, clipnorm=1.0),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      jit_compile="auto" if args.jit else False)
        return model

    def fit(self, hp, model, *a, **kw):
        # F1Val va primero para que el tuner ya vea val_f1_macro en los logs
        kw["callbacks"] = [F1Val()] + list(kw.get("callbacks", []))
        return model.fit(*a, **kw)


class HyperbandPorTandas(kt.Hyperband):
    """Hyperband que sale del proceso (codigo 3) cada N pruebas terminadas.
    on_trial_end de la clase base ya guarda el estado del oracle, asi que el mismo
    comando reanuda exactamente donde quedo, con la GPU limpia."""

    hechas = 0

    def on_trial_end(self, trial):
        super().on_trial_end(trial)
        HyperbandPorTandas.hechas += 1
        if HyperbandPorTandas.hechas >= args.trials_per_process:
            print(f"[{args.trials_per_process} pruebas en este proceso: sale con codigo 3 "
                  f"para liberar la GPU; el mismo comando reanuda]", flush=True)
            sys.exit(3)


tuner = HyperbandPorTandas(
    Modelo(),
    objective=kt.Objective("val_f1_macro", direction="max"),
    max_epochs=args.max_epochs,
    factor=args.factor,
    hyperband_iterations=1,
    seed=SEED,
    directory=str(RUNS),
    project_name=args.model,
    overwrite=False,          # reanuda si existe
)
print(f"== Hyperband {args.model} | max_epochs {args.max_epochs} | factor {args.factor}")
tuner.search_space_summary()
t0 = time.time()
tuner.search(train_ds, epochs=args.max_epochs, verbose=2)
minutos = (time.time() - t0) / 60

# --- resumen ---------------------------------------------------------------------
filas = []
for t in tuner.oracle.trials.values():
    if t.score is None:
        continue
    hpv = t.hyperparameters.values
    filas.append({"trial": t.trial_id, "lr": hpv["lr"], "weight_decay": hpv["weight_decay"],
                  "epocas_asignadas": hpv.get("tuner/epochs"),
                  "mejor_epoca": t.best_step + 1 if t.best_step is not None else None,
                  "val_f1_macro": t.score})
res = pd.DataFrame(filas).sort_values("val_f1_macro", ascending=False)
res.to_csv(OUT / f"{args.model}_trials.csv", index=False)
txt = [f"Nombre: {args.model}_resumen.txt",
       f"Descripcion: Hyperband (keras-tuner) para {args.model}, centralizado, objetivo F1 macro "
       f"en val (195). Generado por 17_hyperband.py v1.4"
       f"{' con --jit (XLA)' if args.jit else ''}. NO usa test.",
       f"Fecha: {time.strftime('%Y-%m-%d')} | {len(res)} pruebas | {minutos:.0f} min "
       f"(si se reanudo, solo cuenta la ultima ejecucion)",
       "", "Mejores 5 pruebas:", res.head(5).to_string(index=False), "",
       "Ojo: val tiene 45 mala; diferencias de F1 menores a ~0,05 son ruido."]
(OUT / f"{args.model}_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")
print("\n" + "\n".join(txt))
