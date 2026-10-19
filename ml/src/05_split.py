"""
Nombre: 05_split.py
Descripcion: Particion sobre los recortes propios (opcion 3). El train del
             equipo (1300) se divide 85/15 estratificado en train/val; el val
             del equipo (334) se conserva intacto como test final.
Fecha_creacion: 2026-10-18
Actualizacion: 2026-10-18
Autor: Julian Andres Silvas Morales
Version: 1.0
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # semilla fija para reproducibilidad
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("ml/results/inventory_crops.csv")
df = df[["subset", "filename", "path", "label", "md5"]].rename(
    columns={"subset": "origen"}
)

team_train = df[df.origen == "train"]
team_val = df[df.origen == "val"]

# 85% train / 15% val, estratificado por clase
train, val = train_test_split(
    team_train, test_size=0.15, stratify=team_train["label"], random_state=SEED
)
test = team_val  # medicion final: no se toca

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.sort_values("path").to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n{name}: {len(part)} recortes")
    print(part["label"].value_counts().to_string())

# Verificacion anti-leakage por ruta y por contenido (md5)
parts = {"train": train, "val": val, "test": test}
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not set(parts[a]["path"]) & set(parts[b]["path"]), f"Ruta repetida {a}-{b}"
    assert not set(parts[a]["md5"]) & set(parts[b]["md5"]), f"Leakage {a}-{b}"
assert sum(len(p) for p in parts.values()) == 1634, "Faltan recortes"
print("\nOK: sin solapamiento y suman 1634")