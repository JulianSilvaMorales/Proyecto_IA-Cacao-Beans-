"""
Nombre: 19_fl_summary.py
Descripcion: Resume el entrenamiento federado formal de los 5 modelos. Lee los
             historiales por ronda de ~/cacao_runs/formal_<modelo>/ (tasa de
             Hyperband) y ~/cacao_runs/formal3_<modelo>/ (tasa alternativa),
             calcula el promedio del F1 macro en val de las rondas 11-15 (criterio
             de eleccion declarado antes de ver los resultados), elige la variante
             ganadora por modelo y copia los historiales a ml/results/federated_formal/.
             No entrena nada y NO usa test.
Fecha_creacion: 2026-10-26
Actualizacion: 2026-10-26
Autor: Julian Andres Silvas Morales (generado con asistencia de Claude)
Version: 1.0
Uso (desde la raiz del repo):
    python ml/src/19_fl_summary.py
"""
import json
from pathlib import Path

import pandas as pd

RUNS = Path.home() / "cacao_runs"
OUT = Path("ml/results/federated_formal")
MODELOS = ["EfficientNet-B0", "MobileNetV3-L", "ResNet-50", "Swin-Tiny", "DeiT-Small"]
# Configuracion de cada variante (la misma que se paso a 10_train_federated.py v1.7):
# 5 clientes, FedAvg, 15 rondas x 1 epoca local, BatchNorm congeladas, clipnorm 1.0.
VARIANTES = {
    "formal": {"EfficientNet-B0": (4.61215120111619e-05, 0.01), "MobileNetV3-L": (9.029419725764854e-05, 0.0),
               "ResNet-50": (9.029419725764854e-05, 0.0), "DeiT-Small": (9.029419725764854e-05, 0.0),
               "Swin-Tiny": (6.230175558672679e-06, 0.0)},
    "formal3": {"EfficientNet-B0": (1.53738373370540e-05, 0.01), "MobileNetV3-L": (3.009806575254951e-05, 0.0),
                "ResNet-50": (3.009806575254951e-05, 0.0), "DeiT-Small": (3.009806575254951e-05, 0.0),
                "Swin-Tiny": (1.869052667601804e-05, 0.0)},
}
RONDAS = range(11, 16)
OUT.mkdir(parents=True, exist_ok=True)

filas = []
for tag, cfg in VARIANTES.items():
    for m in MODELOS:
        h = pd.read_csv(RUNS / f"{tag}_{m}" / "history.csv")
        assert h["ronda"].max() == 15, f"{tag}_{m} no tiene 15 rondas"
        h.to_csv(OUT / f"{tag}_{m}_history.csv", index=False)
        ult = h[h["ronda"].isin(RONDAS)]
        st = json.loads((RUNS / f"{tag}_{m}" / "state.json").read_text())
        filas.append({"modelo": m, "variante": tag, "lr": cfg[m][0], "weight_decay": cfg[m][1],
                      "f1_11_15": ult["f1_macro"].mean(), "f1_min_11_15": ult["f1_macro"].min(),
                      "f1_max_11_15": ult["f1_macro"].max(), "acc_11_15": ult["acc"].mean(),
                      "recall_mala_11_15": ult["recall_mala"].mean(),
                      "f1_ronda15": h.loc[h["ronda"] == 15, "f1_macro"].item(),
                      "mejor_ronda": st["best_round"], "f1_mejor_ronda": st["best_f1"]})

res = pd.DataFrame(filas)
res["elegida"] = res.groupby("modelo")["f1_11_15"].transform("max") == res["f1_11_15"]
res.to_csv(OUT / "fl_resumen.csv", index=False)

ele = res[res["elegida"]].sort_values("f1_11_15", ascending=False)
ver = lambda d: d.round(4).assign(lr=d["lr"].map("{:.2e}".format))   # tasas legibles
txt = ["Nombre: fl_resumen.txt",
       "Descripcion: entrenamiento federado formal (FedAvg, 5 clientes, 15 rondas x 1 epoca local). "
       "Criterio: promedio del F1 macro en val de las rondas 11-15. Se entrega la ronda 15 "
       "(last.weights.h5). Generado por 19_fl_summary.py v1.0. NO usa test.",
       "", "Todas las corridas:",
       ver(res.drop(columns=["elegida"])).to_string(index=False),
       "", "Variante elegida por modelo (se entrega la ronda 15):",
       ver(ele[["modelo", "variante", "lr", "weight_decay", "f1_11_15", "recall_mala_11_15", "f1_ronda15"]])
       .to_string(index=False),
       "", "Ojo: val tiene 45 mala; diferencias de F1 menores a ~0,03-0,05 son ruido."]
(OUT / "fl_resumen.txt").write_text("\n".join(txt) + "\n", encoding="utf8")
(OUT / "modelos_elegidos.json").write_text(json.dumps(
    {r.modelo: {"run_dir": f"~/cacao_runs/{r.variante}_{r.modelo}", "pesos": "last.weights.h5",
                "ronda": 15, "lr": r.lr, "weight_decay": r.weight_decay,
                "jit": r.modelo == "Swin-Tiny"} for r in ele.itertuples()}, indent=2), encoding="utf8")
print("\n".join(txt))
