"""
Nombre: 21_test_final.py
Descripcion: MEDICION FINAL en test (334 granos, 73 mala), que se usa UNA SOLA VEZ.
             Evalua con la ronda 15 del entrenamiento federado formal: los 5 modelos
             elegidos, el ensemble heterogeneo (voto suave, principal, y voto duro),
             el homogeneo de Swin-Tiny y el hibrido de los 10 modelos. Reporta F1
             macro, exactitud, precision y recall por clase, matriz de confusion e
             intervalos de confianza por bootstrap (tabla de D-18 y RNF-07). Mide
             ademas el tiempo de inferencia por grano en CPU (proceso aparte sin GPU,
             con imagenes de val). Se niega a ejecutarse si ya existe un resultado
             de test, para garantizar que test se usa una sola vez.
Fecha_creacion: 2026-10-26
Actualizacion: 2026-10-26
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo):
    python -u ml/src/21_test_final.py --solo-tiempos   # 1.o: tiempos en CPU (no lee test)
    python -u ml/src/21_test_final.py                  # 2.o: medicion final en test (una vez)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--solo-tiempos", action="store_true",
                help="mide el tiempo por grano en CPU con imagenes de val (no lee test)")
ap.add_argument("--hilos", type=int, default=0,
                help="hilos de CPU para --solo-tiempos (0 = todos; 2 aproxima los 2 vCPU de la t4g.medium)")
args = ap.parse_args()
if args.solo_tiempos:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"      # antes de importar TensorFlow: solo CPU
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import importlib
import json
import time

import keras
import keras_hub as kh
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # make_dataset, CLASS_TO_ID

SEED = 42
N_BOOT = 2000
RUNS = Path.home() / "cacao_runs"
OUT = Path("ml/results/test_final")
CACHE = RUNS / "test_final"
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet"),
}
VARIANTES = ("formal", "formal3")
elegidos = json.loads(Path("ml/results/federated_formal/modelos_elegidos.json").read_text())
var_elegida = {m: Path(d["run_dir"]).name.split("_")[0] for m, d in elegidos.items()}


def cargar(m, variante):
    """Modelo con la ronda 15 de ~/cacao_runs/<variante>_<m>/ y su converter."""
    keras.backend.clear_session()
    cls, preset = MODELOS[m]
    _tmp = cls.from_preset(preset, num_classes=2)
    conv = _tmp.preprocessor.image_converter
    del _tmp
    model = cls.from_preset(preset, num_classes=2, activation=None, preprocessor=None)
    model.load_weights(RUNS / f"{variante}_{m}" / "last.weights.h5")
    return model, conv


# --- modo interno: tiempo por grano en CPU (imagenes de val, no de test) ---------
if args.solo_tiempos:
    OUT.mkdir(parents=True, exist_ok=True)
    if args.hilos > 0:
        tf.config.threading.set_intra_op_parallelism_threads(args.hilos)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    print("Dispositivos visibles:", tf.config.list_physical_devices("GPU") or "solo CPU",
          "| hilos:", args.hilos or f"todos ({os.cpu_count()})", flush=True)
    xs = np.concatenate([x.numpy() for x, _ in pipe.make_dataset("val", training=False)])[:96]
    filas = []
    for m in MODELOS:
        model, conv = cargar(m, var_elegida[m])
        xc = conv(tf.constant(xs))
        model.predict(xc[:32], batch_size=32, verbose=0)                      # calentamiento
        t0 = time.perf_counter()
        for _ in range(3):
            model.predict(xc, batch_size=32, verbose=0)
        ms = (time.perf_counter() - t0) / (3 * len(xs)) * 1000
        filas.append({"modelo": m, "hilos": args.hilos or os.cpu_count(), "ms_por_grano_cpu": round(ms, 1),
                      "seg_lote_100_granos": round(ms * 100 / 1000, 2)})
        print(filas[-1], flush=True)
        del model
    t = pd.DataFrame(filas)
    t = pd.concat([t, pd.DataFrame([{"modelo": "ensemble heterogeneo (suma de los 5)",
                                     "hilos": t["hilos"].iloc[0],
                                     "ms_por_grano_cpu": round(t["ms_por_grano_cpu"].sum(), 1),
                                     "seg_lote_100_granos": round(t["seg_lote_100_granos"].sum(), 2)}])])
    t.to_csv(OUT / f"tiempos_cpu_{args.hilos or 'todos'}.csv", index=False)
    sys.exit(0)

# --- proteccion: test se usa UNA sola vez ---------------------------------------------
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
MARCA = OUT / "test_usado.txt"
if MARCA.exists():
    sys.exit(f"Ya existe {MARCA}: test ya se uso ({MARCA.read_text().strip()}) y no se vuelve a medir. "
             f"Si solo fallo la medicion de tiempos: python -u {__file__} --solo-tiempos")
MARCA.write_text(f"test usado el {time.strftime('%Y-%m-%d %H:%M')} por 21_test_final.py v1.0\n",
                 encoding="utf8")

te_df = pd.read_csv("data/splits/test.csv")
y = te_df["label"].map(pipe.CLASS_TO_ID).values
print(f"test: {len(y)} granos | mala: {int(y.sum())} | variante elegida: {var_elegida}", flush=True)

t0 = time.time()
probs = {}
for m in MODELOS:
    for v in VARIANTES:
        model, conv = cargar(m, v)
        ds = pipe.make_dataset("test", training=False).map(lambda a, b: conv(a))
        probs[(m, v)] = tf.nn.softmax(model.predict(ds, verbose=0)).numpy()[:, 1]
        del model
        print(f"   probabilidades de test calculadas: {v} {m}", flush=True)
np.savez_compressed(CACHE / "test_probs.npz", y=y, path=te_df["path"].values,
                    **{f"{v}__{m}": p for (m, v), p in probs.items()})

# --- candidatos (los mismos de 20_ensemble.py) ------------------------------------------
cand = {f"individual | {m} ({var_elegida[m]})": probs[(m, var_elegida[m])] for m in MODELOS}
het = np.stack([probs[(m, var_elegida[m])] for m in MODELOS])
cand["heterogeneo | 5 elegidos, voto suave (PRINCIPAL)"] = het.mean(0)
cand["heterogeneo | 5 elegidos, voto duro"] = ((het >= 0.5).sum(0) >= 3).astype(float)
cand["homogeneo | Swin-Tiny, 2 variantes"] = (probs[("Swin-Tiny", "formal")] + probs[("Swin-Tiny", "formal3")]) / 2
cand["hibrido | 10 modelos (5 arquitecturas x 2 variantes)"] = np.mean(list(probs.values()), axis=0)


def metricas(yt, p):
    pred = (p >= 0.5).astype(int)
    return {"f1_macro": f1_score(yt, pred, average="macro"), "acc": accuracy_score(yt, pred),
            "precision_mala": precision_score(yt, pred, pos_label=1, zero_division=0),
            "recall_mala": recall_score(yt, pred, pos_label=1),
            "precision_sana": precision_score(yt, pred, pos_label=0, zero_division=0),
            "recall_sana": recall_score(yt, pred, pos_label=0)}


rng = np.random.default_rng(SEED)
boot = rng.integers(0, len(y), size=(N_BOOT, len(y)))
filas = []
for nombre, p in cand.items():
    m = metricas(y, p)
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
    b = pd.DataFrame([metricas(y[ix], p[ix]) for ix in boot])
    ic = {f"{k}_ic95": "[{:.3f}, {:.3f}]".format(*np.percentile(b[k], [2.5, 97.5]))
          for k in ("f1_macro", "acc", "recall_mala", "recall_sana")}
    filas.append({"candidato": nombre, **{k: round(v, 4) for k, v in m.items()}, **ic,
                  "VN_sana_ok": tn, "FP_sana_como_mala": fp, "FN_mala_como_sana": fn, "VP_mala_ok": tp})
res = pd.DataFrame(filas)
res.to_csv(OUT / "test_resultados.csv", index=False)
(OUT / "test_metricas.json").write_text(json.dumps(
    {r["candidato"]: {k: r[k] for k in ("f1_macro", "acc", "precision_mala", "recall_mala",
                                        "precision_sana", "recall_sana")} for r in filas},
    indent=2, default=float), encoding="utf8")

pd.set_option("display.width", 250)
cols = ["candidato", "f1_macro", "f1_macro_ic95", "acc", "acc_ic95", "precision_mala", "recall_mala",
        "recall_mala_ic95", "precision_sana", "recall_sana", "VN_sana_ok", "FP_sana_como_mala",
        "FN_mala_como_sana", "VP_mala_ok"]
txt = ["Nombre: test_resumen.txt",
       "Descripcion: medicion final en test (334 granos, 73 mala), usada UNA SOLA VEZ. Modelos del "
       "entrenamiento federado formal (ronda 15) y ensembles. Generado por 21_test_final.py v1.0.",
       f"Fecha: {time.strftime('%Y-%m-%d %H:%M')} | {(time.time() - t0) / 60:.1f} min", "",
       res[cols].to_string(index=False), "",
       "Referencias: predecir siempre sana da exactitud 0,781 en test (261/334) y F1 macro ~0,44.",
       "La meta de trabajo (F1 macro y exactitud de 95 %) NO se compara como cumplida si no se alcanza.",
       "Ensemble principal decidido de antemano: heterogeneo con voto suave."]
(OUT / "test_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")   # se guarda antes de los tiempos
print("\n" + "\n".join(txt))

# --- tiempo por grano en CPU (proceso aparte, sin GPU) ------------------------------------
for h in (0, 2):                                    # normalmente ya se midieron antes de test
    if not (OUT / f"tiempos_cpu_{h or 'todos'}.csv").exists():
        print(f"\nMidiendo el tiempo por grano en CPU, hilos={h or 'todos'} (proceso aparte)...", flush=True)
        subprocess.run([sys.executable, "-u", __file__, "--solo-tiempos", "--hilos", str(h)], check=True)
tiempos = pd.concat([pd.read_csv(OUT / f"tiempos_cpu_{h}.csv") for h in ("todos", 2)])
extra = ["", "Tiempo de inferencia por grano en CPU (CPU de este equipo, NO la t4g.medium de AWS; "
         "2 hilos aproximan sus 2 vCPU; lote de 32, imagenes de val):", tiempos.to_string(index=False)]
with open(OUT / "test_resumen.txt", "a", encoding="utf8") as f:
    f.write("\n".join(extra) + "\n")
print("\n".join(extra))
