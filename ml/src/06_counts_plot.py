"""
Nombre: 06_counts_plot.py
Descripcion: Conteos por clase y particion (train/val/test) a partir de
             data/splits/*.csv. Guarda un CSV de conteos y un grafico de
             barras agrupadas en ml/results/.
Fecha_creacion: 2026-10-18
Actualizacion: 2026-10-18
Autor: Julian Andres Silvas Morales
Version: 1.0
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana: solo guarda el archivo
import matplotlib.pyplot as plt
import pandas as pd

SPLITS = Path("data/splits")
OUT = Path("ml/results")
OUT.mkdir(parents=True, exist_ok=True)

NAMES = ["train", "val", "test"]
CLASSES = ["sana", "mala"]

frames = []
for name in NAMES:
    d = pd.read_csv(SPLITS / f"{name}.csv")
    d["particion"] = name
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

# Tabla de conteos y porcentajes
counts = (
    df.groupby(["particion", "label"]).size().unstack(fill_value=0)
    .reindex(index=NAMES, columns=CLASSES)
)
counts["total"] = counts.sum(axis=1)
counts["pct_mala"] = (counts["mala"] / counts["total"] * 100).round(1)
counts["pct_del_total"] = (counts["total"] / counts["total"].sum() * 100).round(1)
counts.to_csv(OUT / "split_counts.csv")
print(counts.to_string())
print("\nTotal:", int(counts["total"].sum()))

# Grafico de barras agrupadas
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.38
x = range(len(NAMES))
b1 = ax.bar([i - width / 2 for i in x], counts["sana"], width, label="sana")
b2 = ax.bar([i + width / 2 for i in x], counts["mala"], width, label="mala")
ax.bar_label(b1, padding=2)
ax.bar_label(b2, padding=2)
ax.set_xticks(list(x))
ax.set_xticklabels(NAMES)
ax.set_ylabel("Numero de recortes")
ax.set_title("Distribucion de clases por particion (semilla 42)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "split_distribution.png", dpi=150)
print("\nGuardados: split_counts.csv y split_distribution.png en ml/results/")