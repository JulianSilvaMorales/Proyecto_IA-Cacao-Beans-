"""
Nombre: 23_check_duplicados.py
Descripcion: Auditoria de fuga de datos: busca si algun grano de test (334 recortes) es el
             mismo grano que alguno de train + val (1300), aunque sea otra foto. Compara
             (1) caracteristicas de EfficientNet-B0 preentrenado en ImageNet (no los modelos
             del proyecto), probando el recorte de test volteado y girado 180 grados, y
             (2) una huella perceptual (pHash) para fotos repetidas. Calibra con la similitud
             entre granos distintos dentro de train + val, lee la fecha EXIF de las fotos
             originales de los pares mas parecidos y guarda una cuadricula para revisarlos a
             ojo. No mide ningun modelo. Con --excluir recalcula las metricas de test del
             modelo de la app quitando los granos marcados como repetidos, a partir de las
             probabilidades ya guardadas por 21_test_final.py (no vuelve a predecir).
Fecha_creacion: 2026-11-25
Actualizacion: 2026-11-25
Autor: Julian Andres Silvas Morales
Version: 1.1
Uso (desde la raiz del repo):
    python -u ml/src/23_check_duplicados.py
    python -u ml/src/23_check_duplicados.py --excluir ml/results/duplicados/repetidos.txt
"""
import argparse
import importlib
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras_hub as kh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.fft import dct
from sklearn.metrics import accuracy_score, f1_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # make_dataset, load_image

OUT = Path("ml/results/duplicados")
ORIG = Path("data/raw/dataset_original_train_val")
PRESET = "efficientnet_b0_ra4_e3600_r224_imagenet"
N_CSV, N_GRID = 50, 30
ap = argparse.ArgumentParser()
ap.add_argument("--excluir", default=None,
                help="txt con rutas de test (una por linea) marcadas como repetidas tras la revision a ojo")
args = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)

ref = pd.concat([pd.read_csv("data/splits/train.csv").assign(conjunto="train"),
                 pd.read_csv("data/splits/val.csv").assign(conjunto="val")], ignore_index=True)
te = pd.read_csv("data/splits/test.csv").assign(conjunto="test")
assert len(ref) == 1300 and len(te) == 334

# --- modo recalculo: metricas de test sin los granos repetidos ------------------------------
if args.excluir:
    quitar = {l.strip() for l in Path(args.excluir).read_text(encoding="utf8").splitlines() if l.strip()}
    assert quitar <= set(te["path"]), "hay rutas que no son de test"
    d = np.load(Path.home() / "cacao_runs/test_final/test_probs.npz", allow_pickle=True)
    assert (d["path"] == te["path"].values).all()
    y, p = d["y"], d["formal3__EfficientNet-B0"]
    keep = ~np.isin(te["path"].values, list(quitar))
    filas = []
    for nombre, m in [("todos los granos de test", np.ones(len(y), bool)), ("sin los repetidos", keep)]:
        pred = (p[m] >= 0.5).astype(int)
        filas.append({"conjunto": nombre, "granos": int(m.sum()), "mala": int(y[m].sum()),
                      "f1_macro": round(f1_score(y[m], pred, average="macro"), 4),
                      "acc": round(accuracy_score(y[m], pred), 4),
                      "recall_mala": round(recall_score(y[m], pred, pos_label=1), 4),
                      "recall_sana": round(recall_score(y[m], pred, pos_label=0), 4)})
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "test_sin_repetidos.csv", index=False)
    print(f"Modelo de la app (EfficientNet-B0), {len(quitar)} granos de test excluidos:\n{r.to_string(index=False)}")
    sys.exit(0)

t0 = time.time()


# --- 1) caracteristicas de EfficientNet-B0 (ImageNet) ----------------------------------------
def cargar(paths):
    ds = tf.data.Dataset.from_tensor_slices((paths, np.zeros(len(paths), "int32")))
    ds = ds.map(pipe.load_image, num_parallel_calls=tf.data.AUTOTUNE).map(lambda a, b: a).batch(64)
    return np.concatenate([b.numpy() for b in ds])


_tmp = kh.models.EfficientNetImageClassifier.from_preset(PRESET)
conv = _tmp.preprocessor.image_converter
del _tmp
backbone = kh.models.EfficientNetBackbone.from_preset(PRESET)


def emb(x, lote=16):
    """Lotes pequenos: en modo eager las activaciones intermedias ocupan mucha memoria de GPU."""
    out = []
    for i in range(0, len(x), lote):
        f = backbone(conv(tf.constant(x[i:i + lote])), training=False)
        out.append(tf.reduce_mean(f, axis=[1, 2]).numpy())
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


x_ref, x_te = cargar(ref["path"].values), cargar(te["path"].values)
e_ref = emb(x_ref)
# el recorte deja el eje largo horizontal: el mismo grano puede aparecer volteado o girado 180
vistas = {"original": x_te, "volteo_h": x_te[:, :, ::-1], "volteo_v": x_te[:, ::-1],
          "giro_180": x_te[:, ::-1, ::-1]}
sims = {k: emb(np.ascontiguousarray(v)) @ e_ref.T for k, v in vistas.items()}       # (334, 1300)
sim = np.max(np.stack(list(sims.values())), axis=0)
vista = np.array(list(sims))[np.argmax(np.stack(list(sims.values())), axis=0)]

# calibracion: vecino mas parecido entre granos DISTINTOS (dentro de train + val)
s_rr = e_ref @ e_ref.T
np.fill_diagonal(s_rr, -1)
nn_ref = s_rr.max(1)


# --- 2) huella perceptual (pHash 64 bits) ------------------------------------------------------
def phash(img):
    g = np.asarray(Image.fromarray(img.astype("uint8")).convert("L").resize((32, 32), Image.LANCZOS), float)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].ravel()[1:]
    return d > np.median(d)


h_ref = np.array([phash(i) for i in x_ref])
h_te = {k: np.array([phash(i) for i in v]) for k, v in vistas.items()}
ham = np.min(np.stack([(h[:, None, :] != h_ref[None, :, :]).sum(-1) for h in h_te.values()]), axis=0)
# calibracion del pHash: distancia al mas cercano entre granos de train+val
ham_rr = (h_ref[:, None, :] != h_ref[None, :, :]).sum(-1)
np.fill_diagonal(ham_rr, 99)
nn_ham_ref = ham_rr.min(1)


# --- 3) fecha EXIF de la foto original --------------------------------------------------------
def exif_fecha(fila):
    p = ORIG / fila["origen"] / fila["filename"]
    try:
        with Image.open(p) as im:
            v = im.getexif().get_ifd(0x8769).get(36867) or im.getexif().get(306)
        return datetime.strptime(v, "%Y:%m:%d %H:%M:%S") if v else None
    except Exception:  # noqa: BLE001
        return None


# --- pares mas parecidos (para cada grano de test, su vecino mas parecido) --------------------
j = sim.argmax(1)
pares = pd.DataFrame({"test_path": te["path"].values, "test_label": te["label"].values,
                      "ref_path": ref["path"].values[j], "ref_conjunto": ref["conjunto"].values[j],
                      "ref_label": ref["label"].values[j], "similitud": sim[np.arange(len(te)), j].round(4),
                      "vista": vista[np.arange(len(te)), j],
                      "phash_dist_min": ham.min(1), "phash_ref_mas_cercano": ref["path"].values[ham.argmin(1)]})
pares = pares.sort_values("similitud", ascending=False).reset_index(drop=True)
top = pares.head(N_CSV).copy()
top["test_fecha"] = [exif_fecha(te.set_index("path").loc[p]) for p in top["test_path"]]
top["ref_fecha"] = [exif_fecha(ref.set_index("path").loc[p]) for p in top["ref_path"]]
top["seg_entre_fotos"] = [abs((a - b).total_seconds()) if a is not None and b is not None else None
                          for a, b in zip(top["test_fecha"], top["ref_fecha"])]
top.to_csv(OUT / "pares_top.csv", index=False)
n_exif = sum(f is not None for f in top["test_fecha"])

# cuadricula para revisar a ojo
fig, axes = plt.subplots(N_GRID // 3, 6, figsize=(18, 3.1 * N_GRID // 3))
for k, fila in top.head(N_GRID).iterrows():
    for c, (ruta, et) in enumerate([(fila["test_path"], f"#{k + 1} TEST {fila['test_label']}"),
                                    (fila["ref_path"], f"{fila['ref_conjunto'].upper()} {fila['ref_label']} "
                                                       f"sim {fila['similitud']:.3f}")]):
        ax = axes[k // 3, (k % 3) * 2 + c]
        ax.imshow(Image.open(ruta).convert("RGB"))
        ax.set_title(f"{et}\n{Path(ruta).parent.name}/{Path(ruta).name}", fontsize=7)
        ax.axis("off")
fig.suptitle("Pares test / train+val mas parecidos (izquierda: test; derecha: su vecino mas parecido)", y=1.0)
fig.tight_layout()
fig.savefig(OUT / "pares_top.png", dpi=70)
plt.close(fig)

# resumen
pct = lambda a, q: float(np.percentile(a, q))
txt =["Nombre: duplicados_resumen.txt",
       "Descripcion: auditoria de fuga de datos test vs train+val (23_check_duplicados.py v1.0). "
       "No mide modelos.",
       f"Fecha: {time.strftime('%Y-%m-%d %H:%M')} | {(time.time() - t0) / 60:.1f} min", "",
       "Similitud coseno (caracteristicas EfficientNet-B0 ImageNet; maximo sobre original, volteos y giro 180):",
       f"  test -> vecino mas parecido en train+val: mediana {np.median(sim.max(1)):.3f} | "
       f"p99 {pct(sim.max(1), 99):.3f} | maximo {sim.max():.3f}",
       f"  calibracion, granos distintos dentro de train+val -> vecino mas parecido: mediana {np.median(nn_ref):.3f} | "
       f"p99 {pct(nn_ref, 99):.3f} | maximo {nn_ref.max():.3f}",
       f"  granos de test por encima del maximo de la calibracion ({nn_ref.max():.3f}): "
       f"{int((sim.max(1) > nn_ref.max()).sum())}", "",
       "Huella perceptual (pHash, distancia de Hamming sobre 64 bits al recorte mas cercano):",
       f"  test -> train+val: mediana {np.median(ham.min(1)):.0f} | minima {int(ham.min())} | "
       f"granos de test con distancia <= 5: {int((ham.min(1) <= 5).sum())}",
       f"  calibracion, granos de train+val entre si: mediana {np.median(nn_ham_ref):.0f} | minima {int(nn_ham_ref.min())} "
       f"| con distancia <= 5: {int((nn_ham_ref <= 5).sum())} de {len(nn_ham_ref)}",
       "  Si la calibracion tambien da distancias pequenas, el pHash no distingue granos en estas imagenes "
       "(todas son un ovalo oscuro sobre papel gris) y no sirve como prueba de duplicado.", "",
       f"EXIF: fecha disponible en {n_exif} de {len(top)} fotos de test de la tabla "
       f"({'las fotos originales no traen fecha: no se puede usar' if n_exif == 0 else 'ver pares_top.csv'}).",
       "Pares para revisar a ojo: ml/results/duplicados/pares_top.png (30) y pares_top.csv (50).",
       "Un par es el mismo grano si coinciden forma, grietas y manchas en el mismo lugar (tambien volteado)."]
(OUT / "duplicados_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")
print("\n".join(txt))
print("\nTop 10 pares:\n" + top.head(10)[["test_path", "ref_path", "ref_conjunto", "similitud", "vista",
                                          "phash_dist_min", "seg_entre_fotos"]].to_string(index=False))
