"""
Nombre: 04_inventory_crops.py
Descripcion: Inventario de los recortes propios (data/processed/recortes_script):
             conteos por carpeta y clase, integridad, duplicados por md5,
             tamanos y cruce con log_train.csv / log_val.csv.
Fecha_creacion: 2026-10-18
Actualizacion: 2026-10-18
Autor: Julian Andres Silvas Morales
Version: 1.0
"""
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image

BASE = Path("data/processed/recortes_script")
OUT = Path("ml/results/inventory_crops.csv")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def label_from_name(name: str) -> str:
    """'mala (12).jpg' -> 'mala'; 'sana (5).jpg' -> 'sana'."""
    m = re.match(r"([a-zA-Z]+)", name)
    return m.group(1).lower() if m else "desconocida"


rows = []
for subset in ["train", "val"]:
    for p in sorted((BASE / subset).glob("*.jpg")):
        row = {
            "subset": subset,
            "filename": p.name,
            "path": p.as_posix(),
            "label": label_from_name(p.name),
            "md5": md5(p),
            "corrupt": False,
            "width": None,
            "height": None,
        }
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                row["width"], row["height"] = im.size
        except Exception:
            row["corrupt"] = True
        rows.append(row)

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Total recortes:", len(df))
print("\nPor carpeta y clase:\n", df.groupby(["subset", "label"]).size())
print("\nCorruptos:", int(df["corrupt"].sum()))
print("Duplicados por md5 (todo el conjunto):", int(df.duplicated("md5").sum()))

# Mismo contenido en train y val (leakage)
tr = set(df[df.subset == "train"]["md5"])
va = set(df[df.subset == "val"]["md5"])
print("Archivos con igual contenido en train y val:", len(tr & va))

print("\nTamano (ancho x alto), resumen:")
print(df[["width", "height"]].describe().loc[["min", "50%", "max"]])

# Cruce con los logs del companero
for subset in ["train", "val"]:
    log = pd.read_csv(BASE / f"log_{subset}.csv")
    sub = df[df.subset == subset]
    print(f"\nlog_{subset}.csv: {len(log)} filas")
    print("  En log y sin recorte en disco:",
          len(set(log["archivo"]) - set(sub["filename"])))
    print("  En disco y sin fila en log:",
          len(set(sub["filename"]) - set(log["archivo"])))
    print("  Notas del log:\n", log["nota"].fillna("(sin nota)").value_counts().to_string())