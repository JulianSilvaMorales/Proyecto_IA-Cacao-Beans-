"""
Nombre: 22_export.py
Descripcion: Exporta los 5 modelos del entrenamiento federado formal (ronda 15) para
             la app. Cada modelo exportado recibe la imagen del grano ya rellenada y
             redimensionada a 224x224 (RGB, 0-255) y devuelve [P(sana), P(mala)]: la
             escala del preset y el softmax van dentro. Formatos: .keras (los 5) y
             SavedModel de TensorFlow (modelo de la app, no necesita keras-hub). Escribe
             un JSON por modelo (entrada, clases, umbral, metricas de test y val,
             tiempos, correccion del lote, versiones, sha256), un inferencia.py de
             ejemplo y un manifest. Verifica que cada archivo recargado da las mismas
             probabilidades que el modelo original en val. NO vuelve a usar test (lee
             sus metricas ya guardadas).
Fecha_creacion: 2026-10-26
Actualizacion: 2026-10-26
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo):
    python -u ml/src/22_export.py
"""
import hashlib
import importlib
import json
import os
import shutil
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

sys.path.insert(0, str(Path(__file__).parent))
pipe = importlib.import_module("07_data_pipeline")   # make_dataset, load_image

VERSION = "1.0"
RUNS = Path.home() / "cacao_runs"
EXPORT = Path.home() / "cacao_export" / f"modelos-v{VERSION}"     # pesado: fuera del repo
OUT = Path("ml/results/export")                                     # manifest liviano (versionado)
MODELO_APP = "EfficientNet-B0"
MODELOS = {
    "MobileNetV3-L": (kh.models.MobileNetImageClassifier, "mobilenet_v3_large_100_imagenet", "CNN"),
    "EfficientNet-B0": (kh.models.EfficientNetImageClassifier, "efficientnet_b0_ra4_e3600_r224_imagenet", "CNN"),
    "ResNet-50": (kh.models.ResNetImageClassifier, "resnet_50_imagenet", "CNN"),
    "Swin-Tiny": (kh.models.SwinTransformerImageClassifier, "swin_tiny_patch4_window7_224", "Transformer"),
    "DeiT-Small": (kh.models.DeiTImageClassifier, "deit_small_distilled_patch16_224_imagenet", "Transformer"),
}
elegidos = json.loads(Path("ml/results/federated_formal/modelos_elegidos.json").read_text())
fl = pd.read_csv("ml/results/federated_formal/fl_resumen.csv")
test_m = json.loads(Path("ml/results/test_final/test_metricas.json").read_text())
tiempos = pd.concat([pd.read_csv(f"ml/results/test_final/tiempos_cpu_{h}.csv") for h in ("todos", 2)])
assert Path("ml/results/test_final/test_usado.txt").exists(), "falta la medicion final en test"

if EXPORT.exists():
    shutil.rmtree(EXPORT)                     # se regenera completo (archivos derivados)
EXPORT.mkdir(parents=True)
OUT.mkdir(parents=True, exist_ok=True)
xs_val = np.concatenate([x.numpy() for x, _ in pipe.make_dataset("val", training=False)])


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def construir(m):
    """Modelo exportable: entrada 224x224 RGB 0-255 -> escala del preset -> red -> softmax."""
    cls, preset, _ = MODELOS[m]
    _tmp = cls.from_preset(preset, num_classes=2)
    conv = _tmp.preprocessor.image_converter
    scale = [float(v) for v in np.ravel(conv.scale)]
    offset = [float(v) for v in np.ravel(conv.offset)]
    del _tmp
    base = cls.from_preset(preset, num_classes=2, activation=None, preprocessor=None)
    base.load_weights(Path(elegidos[m]["run_dir"].replace("~", str(Path.home()))) / elegidos[m]["pesos"])
    inp = keras.Input((224, 224, 3), name="imagen_0_255")
    x = keras.layers.Rescaling(scale=scale, offset=offset, name="escala_del_preset")(inp)
    out = keras.layers.Softmax(name="probabilidades")(base(x))
    return keras.Model(inp, out, name=f"inspectia_{m.replace('-', '_')}"), scale, offset


def equivalencia(p, q):
    """Grafos distintos (predict compilado, tf.data, SavedModel) redondean distinto (~1e-3 en
    probabilidad): se exige que NINGUNA clase cambie y que la diferencia sea pequena."""
    eq = {"max_dif_prob": float(np.abs(p - q).max()), "cambios_de_clase": int(((p >= 0.5) != (q >= 0.5)).sum())}
    assert eq["cambios_de_clase"] == 0 and eq["max_dif_prob"] < 5e-3, f"no equivale: {eq}"
    return eq


manifest = {"version": VERSION, "fecha": time.strftime("%Y-%m-%d"), "modelo_app": MODELO_APP, "archivos": {}}
for m, (cls, preset, tipo) in MODELOS.items():
    keras.backend.clear_session()
    print(f"\n== {m}", flush=True)
    modelo, scale, offset = construir(m)
    p_env = modelo.predict(xs_val, batch_size=32, verbose=0)[:, 1]
    ref = p_env                                   # referencia del archivo exportado (mismo grafo)
    # equivalencia con las probabilidades con las que se reportaron val y test (20/21)
    ref_pipe = np.load(RUNS / "ensemble" / f"val_{Path(elegidos[m]['run_dir']).name.split('_')[0]}_{m}.npy")
    eq = equivalencia(p_env, ref_pipe)
    print(f"   exportable vs probabilidades de 20/21: max |dif| = {eq['max_dif_prob']:.2e} | "
          f"cambios de clase en val: {eq['cambios_de_clase']}", flush=True)

    nombre = f"inspectia_{m}_v{VERSION}"
    f_keras = EXPORT / f"{nombre}.keras"
    modelo.save(f_keras)
    keras.backend.clear_session()
    recargado = keras.models.load_model(f_keras)
    d = np.abs(recargado.predict(xs_val, batch_size=32, verbose=0)[:, 1] - ref).max()
    print(f"   .keras recargado vs original: max |dif| = {d:.2e} | {f_keras.stat().st_size / 2**20:.1f} MB", flush=True)
    assert d < 1e-5, "el .keras recargado no reproduce al original"
    archivos = {f_keras.name: sha256(f_keras)}

    if m == MODELO_APP:
        d_sm = EXPORT / f"{nombre}_savedmodel"
        recargado.export(str(d_sm), format="tf_saved_model")
        sm = tf.saved_model.load(str(d_sm))
        p_sm = np.concatenate([sm.serve(tf.constant(xs_val[i:i + 32])).numpy()[:, 1]
                               for i in range(0, len(xs_val), 32)])
        eq_sm = equivalencia(p_sm, ref)
        print(f"   SavedModel vs .keras: max |dif| = {eq_sm['max_dif_prob']:.2e} | "
              f"cambios de clase: {eq_sm['cambios_de_clase']}", flush=True)
        z = shutil.make_archive(str(d_sm), "zip", root_dir=EXPORT, base_dir=d_sm.name)
        shutil.rmtree(d_sm)
        archivos[Path(z).name] = sha256(z)

    fila = fl[(fl["modelo"] == m) & (fl["elegida"])].iloc[0]
    tm = test_m[next(k for k in test_m if k.startswith("individual") and f"| {m} (" in k)]
    t = tiempos[tiempos["modelo"] == m].set_index("hilos")["ms_por_grano_cpu"].to_dict()
    ficha = {
        "nombre": nombre, "version": VERSION, "es_modelo_app": m == MODELO_APP,
        "arquitectura": m, "tipo": tipo, "preset_keras_hub": preset,
        "entrada": {"forma": [None, 224, 224, 3], "tipo": "float32", "rango": "0-255", "canales": "RGB",
                    "preparacion": ("Recorte de UN grano (generar_recortes.py). Volverlo cuadrado sin deformarlo "
                                    "prolongando su borde (promedio de 4 filas/columnas, suavizado a 16 valores) y "
                                    "redimensionar a 224x224 (bilineal). Ver inferencia.py (igual a "
                                    "07_data_pipeline.load_image)."),
                    "escala_del_preset_incluida": {"scale": scale, "offset": offset}},
        "salida": {"forma": [None, 2], "significado": "[P(sana), P(mala)]", "suma": 1.0},
        "clases": {"0": "sana", "1": "mala"}, "umbral_mala": 0.5,
        "entrenamiento": {"metodo": "Federated Learning, FedAvg propio, 5 clientes estratificados",
                          "rondas": 15, "epocas_locales": 1, "lr": float(fila["lr"]),
                          "weight_decay": float(fila["weight_decay"]), "batchnorm": "congeladas",
                          "clipnorm": 1.0, "pesos": "ronda 15 (last.weights.h5)",
                          "variante": elegidos[m]["run_dir"].split("/")[-1].split("_")[0],
                          "preentrenado": "ImageNet"},
        "metricas_test": {k: round(float(v), 4) for k, v in tm.items()},
        "verificacion_exportacion": {"keras_recargado_vs_exportado": "max |dif| < 1e-5 en val (mismo grafo)",
                                     "frente_a_las_probabilidades_reportadas_val": eq,
                                     "nota": ("grafos distintos (tf.data, predict compilado, SavedModel) redondean "
                                              "distinto, ~1e-3 en probabilidad; se exige 0 cambios de clase")},
        "metricas_val_rondas_11_15": {"f1_macro": round(float(fila["f1_11_15"]), 4),
                                      "recall_mala": round(float(fila["recall_mala_11_15"]), 4)},
        "tiempo_cpu_ms_por_grano": {"12_hilos": t.get(12), "2_hilos_aprox_t4g_medium": t.get(2),
                                    "nota": "CPU de desarrollo, no la t4g.medium de AWS"},
        "correccion_porcentaje_lote": {
            "formula": "p_real = (p_obs - (1 - recall_mala)) / (recall_sana - (1 - recall_mala)), limitada a [0, 1]",
            "recall_sana": round(float(tm["recall_sana"]), 4), "recall_mala": round(float(tm["recall_mala"]), 4),
            "advertencia": "amplifica el error de muestreo por 1/(recall_sana + recall_mala - 1)"},
        "versiones": {"tensorflow": tf.__version__, "keras": keras.__version__, "keras_hub": kh.__version__},
        "dataset": "Ecuadorian cacao beans for classification (Kaggle), CC BY-NC 4.0: uso no comercial",
        "archivos_sha256": archivos,
    }
    (EXPORT / f"{nombre}.json").write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf8")
    manifest["archivos"].update(archivos)
    manifest["archivos"][f"{nombre}.json"] = sha256(EXPORT / f"{nombre}.json")
    del modelo, recargado

# --- ejemplo de inferencia autocontenido para el servicio de IA -------------------------
ejemplo = '''"""
Nombre: inferencia.py
Descripcion: Ejemplo de inferencia con un modelo exportado de InspectIA (release modelos-v{v}).
             Prepara un recorte de UN grano igual que el entrenamiento y devuelve P(mala).
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: {v}
Uso:
    python inferencia.py inspectia_EfficientNet-B0_v{v}.keras recorte.jpg
    (requiere tensorflow 2.21 y, para .keras, keras-hub 0.32.0; el SavedModel solo tensorflow)
"""
import sys

import tensorflow as tf


def _suavizar(fila, puntos=16):
    n = tf.shape(fila)[1]
    corta = tf.image.resize(fila, (1, puntos), method="area")
    return tf.image.resize(corta, (1, n), method="bilinear")


def preparar(ruta_jpg):
    """Igual a 07_data_pipeline.load_image: cuadrado prolongando el borde, 224x224, 0-255."""
    img = tf.cast(tf.io.decode_jpeg(tf.io.read_file(ruta_jpg), channels=3), tf.float32)
    h, w = tf.shape(img)[0], tf.shape(img)[1]
    lado = tf.maximum(h, w)
    top, left = (lado - h) // 2, (lado - w) // 2
    bottom, right = lado - h - top, lado - w - left
    k = 4
    img = tf.concat([tf.repeat(_suavizar(tf.reduce_mean(img[:k], 0, keepdims=True)), top, 0), img,
                     tf.repeat(_suavizar(tf.reduce_mean(img[-k:], 0, keepdims=True)), bottom, 0)], 0)
    ci = tf.transpose(_suavizar(tf.transpose(tf.reduce_mean(img[:, :k], 1, keepdims=True), [1, 0, 2])), [1, 0, 2])
    cd = tf.transpose(_suavizar(tf.transpose(tf.reduce_mean(img[:, -k:], 1, keepdims=True), [1, 0, 2])), [1, 0, 2])
    img = tf.concat([tf.repeat(ci, left, 1), img, tf.repeat(cd, right, 1)], 1)
    return tf.clip_by_value(tf.image.resize(img, (224, 224)), 0.0, 255.0)


if __name__ == "__main__":
    ruta_modelo, ruta_img = sys.argv[1], sys.argv[2]
    x = preparar(ruta_img)[None]
    if ruta_modelo.endswith(".keras"):
        import keras
        import keras_hub  # noqa: F401  (registra las capas del modelo)
        p = keras.models.load_model(ruta_modelo).predict(x, verbose=0)[0]
    else:                                   # carpeta del SavedModel descomprimida
        sm = tf.saved_model.load(ruta_modelo)   # guardar la referencia: si no, se libera antes de usarlo
        p = sm.serve(x).numpy()[0]
    print(f"P(sana) = {{p[0]:.3f} | P(mala) = {{p[1]:.3f} -> {{'mala' if p[1] >= 0.5 else 'sana'}}")
'''.replace("{v}", VERSION).replace("{{", "{").replace("}}", "}")
(EXPORT / "inferencia.py").write_text(ejemplo, encoding="utf8")
manifest["archivos"]["inferencia.py"] = sha256(EXPORT / "inferencia.py")

# comprobacion: el preparar() del ejemplo es identico al del pipeline
ns = {}
exec(compile(ejemplo, "inferencia.py", "exec"), ns)
p0 = pipe.make_dataset("val", training=False).unbatch().take(3)
ruta0 = pd.read_csv("data/splits/val.csv")["path"].values[:3]
dif = max(float(np.abs(ns["preparar"](r).numpy() - x.numpy()).max()) for r, (x, _) in zip(ruta0, p0))
print(f"\npreparar() de inferencia.py vs pipeline: max |dif| = {dif:.2e}")
assert dif < 1e-3, "inferencia.py no prepara la imagen igual que el pipeline"

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
shutil.copy(OUT / "manifest.json", EXPORT / "manifest.json")
print(f"\nExportado en {EXPORT}:")
for f in sorted(EXPORT.iterdir()):
    print(f"   {f.name:<55} {f.stat().st_size / 2**20:8.1f} MB")
