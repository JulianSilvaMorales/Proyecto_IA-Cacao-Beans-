"""
Nombre: 03_split.py
Descripcion: Particion estratificada 70/15/15 (train/val/test) de las
             imagenes de dataset_cropped_allclasses. Guarda CSV en data/splits.
Fecha_creacion: 2026-10-10
Actualizacion: 2026-10-10
Version: 1.0
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # fija la semilla para que la particion sea reproducible
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("ml/results/inventory.csv")
df = df[df.source == "cropped"][["path", "filename", "label", "md5"]].copy()
df["path"] = df["path"].str.replace("\\", "/", regex=False)

# 70% train, 30% restante
train, rest = train_test_split(
    df, test_size=0.30, stratify=df["label"], random_state=SEED
)
# el 30% se divide en mitades: 15% val, 15% test
val, test = train_test_split(
    rest, test_size=0.50, stratify=rest["label"], random_state=SEED
)

for name, part in [("train", train), ("val", val), ("test", test)]:
    part.sort_values("filename").to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n{name}: {len(part)} imagenes")
    print(part["label"].value_counts().to_string())

# Verificacion anti-leakage: ningun archivo en mas de una particion
sets = {n: set(p["md5"]) for n, p in [("train", train), ("val", val), ("test", test)]}
assert not (sets["train"] & sets["val"]), "Leakage train-val"
assert not (sets["train"] & sets["test"]), "Leakage train-test"
assert not (sets["val"] & sets["test"]), "Leakage val-test"
print("\nOK: sin solapamiento entre particiones")