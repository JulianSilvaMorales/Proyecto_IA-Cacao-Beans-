"""
Nombre: 20_ensemble.py
Descripcion: Ensembles (Topic 5) con los modelos del entrenamiento federado formal,
             evaluados SOLO en val. Carga la ronda 15 (last.weights.h5) de las 10
             corridas (5 arquitecturas x 2 variantes de tasa), calcula P(mala) en
             val y compara: modelos individuales; ensemble heterogeneo (5 elegidos,
             voto suave y voto duro); homogeneo (misma arquitectura, sus 2
             variantes); hibrido (los 10). Intervalos por bootstrap y diferencia
             pareada frente al mejor modelo individual. No entrena y NO usa test.
Fecha_creacion: 2026-10-26
Actualizacion: 2026-10-26
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo):
    python -u ml/src/20_ensemble.py
"""
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
N_BOOT = 2000
RUNS = Path.home() / "cacao_runs"
OUT = Path("ml/results/ensemble")
CACHE = RUNS / "ensemble"
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
}
VARIANTES = ("formal", "formal3")
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

elegidos = json.loads(Path("ml/results/federated_formal/modelos_elegidos.json").read_text())
var_elegida = {m: Path(d["run_dir"]).name.split("_")[0] for m, d in elegidos.items()}
print("Variante elegida por modelo:", var_elegida)

va_df = pd.read_csv("data/splits/val.csv")
y = va_df["label"].map(pipe.CLASS_TO_ID).values


def p_mala(m, variante):
    """P(mala) en val con la ronda 15 de ~/cacao_runs/<variante>_<m>/ (con cache)."""
    f = CACHE / f"val_{variante}_{m}.npy"
    if f.exists():
        return np.load(f)
    keras.backend.clear_session()
    cls, preset = MODELOS[m]
    _tmp = cls.from_preset(preset, num_classes=2)
    conv = _tmp.preprocessor.image_converter
    del _tmp
    model = cls.from_preset(preset, num_classes=2, activation=None, preprocessor=None)
    model.load_weights(RUNS / f"{variante}_{m}" / "last.weights.h5")
    ds = pipe.make_dataset("val", training=False).map(lambda a, b: conv(a))
    p = tf.nn.softmax(model.predict(ds, verbose=0)).numpy()[:, 1]
    np.save(f, p)
    del model
    return p


def metricas(yt, p, umbral=0.5):
    pred = (p >= umbral).astype(int)
    return {"f1_macro": f1_score(yt, pred, average="macro"), "acc": accuracy_score(yt, pred),
            "recall_mala": recall_score(yt, pred, pos_label=1),
            "recall_sana": recall_score(yt, pred, pos_label=0)}


t0 = time.time()
probs = {}
for m in MODELOS:
    for v in VARIANTES:
        probs[(m, v)] = p_mala(m, v)
        print(f"{v:<8} {m:<16} F1 val (ronda 15) {metricas(y, probs[(m, v)])['f1_macro']:.3f}", flush=True)

# --- candidatos -------------------------------------------------------------------
cand = {}
for m in MODELOS:
    cand[f"individual | {m} ({var_elegida[m]})"] = probs[(m, var_elegida[m])]
het = np.stack([probs[(m, var_elegida[m])] for m in MODELOS])
cand["heterogeneo | 5 elegidos, voto suave (PRINCIPAL)"] = het.mean(0)
cand["heterogeneo | 5 elegidos, voto duro"] = ((het >= 0.5).sum(0) >= 3).astype(float)
for m in MODELOS:
    cand[f"homogeneo | {m}, 2 variantes"] = (probs[(m, "formal")] + probs[(m, "formal3")]) / 2
cand["hibrido | 10 modelos (5 arquitecturas x 2 variantes)"] = np.mean(list(probs.values()), axis=0)

# referencia de la comparacion pareada: mejor individual (por F1 en val, ronda 15)
indiv = {k: v for k, v in cand.items() if k.startswith("individual")}
ref = max(indiv, key=lambda k: metricas(y, indiv[k])["f1_macro"])
rng = np.random.default_rng(SEED)
boot = rng.integers(0, len(y), size=(N_BOOT, len(y)))
filas = []
for nombre, p in cand.items():
    m = metricas(y, p)
    b = np.array([[f1_score(y[ix], (p[ix] >= 0.5).astype(int), average="macro"),
                   f1_score(y[ix], (cand[ref][ix] >= 0.5).astype(int), average="macro")] for ix in boot])
    lo, hi = np.percentile(b[:, 0], [2.5, 97.5])
    dlo, dhi = np.percentile(b[:, 0] - b[:, 1], [2.5, 97.5])
    filas.append({"candidato": nombre, **{k: round(v, 4) for k, v in m.items()},
                  "f1_ic95": f"[{lo:.3f}, {hi:.3f}]",
                  "dif_f1_vs_mejor_individual": round(m["f1_macro"] - metricas(y, cand[ref])["f1_macro"], 4),
                  "dif_ic95": f"[{dlo:+.3f}, {dhi:+.3f}]"})
res = pd.DataFrame(filas)
res.to_csv(OUT / "ensemble_val.csv", index=False)
np.savez_compressed(CACHE / "val_candidatos.npz", y=y, **{k: v for k, v in cand.items()})

pd.set_option("display.width", 250)
txt = ["Nombre: ensemble_val_resumen.txt",
       "Descripcion: ensembles con los modelos federados formales (ronda 15), evaluados SOLO en val "
       "(195 granos, 45 mala). Generado por 20_ensemble.py v1.0. NO usa test.",
       f"Fecha: {time.strftime('%Y-%m-%d')} | {(time.time() - t0) / 60:.1f} min",
       f"Referencia de la diferencia pareada: {ref}", "",
       res.to_string(index=False), "",
       "Ojo: val ya se uso para elegir la variante y la ronda de cada modelo: estas cifras son algo",
       "optimistas y diferencias de F1 menores a ~0,05 son ruido. La comparacion definitiva sale de test.",
       "Se decidio de antemano llevar a test como ensemble principal el heterogeneo con voto suave."]
(OUT / "ensemble_val_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")
print("\n" + "\n".join(txt))
