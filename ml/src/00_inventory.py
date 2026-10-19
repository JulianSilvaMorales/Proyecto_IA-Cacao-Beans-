"""
Nombre: 00_inventory.py
Descripcion: Inventario del dataset crudo: ruta, etiqueta (extraida del
             nombre), tamaño, integridad y hash para detectar duplicados.
Fecha_creacion: 2026-10-09
Actualizacion: 2026-10-09
Autor: Julian Andres Silvas Morales
Version: 1.1
"""
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path("data/raw")
OUT = Path("ml/results/inventory.csv")
EXTS = {".jpg", ".jpeg", ".png"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def label_from_name(name: str) -> str:
    """'mala (12).jpg' -> 'mala'; 'sana (5).jpg' -> 'sana'."""
    m = re.match(r"([a-zA-Z]+)", name)
    return m.group(1).lower() if m else "desconocida"


rows = []
for p in sorted(RAW.rglob("*")):
    if p.suffix.lower() not in EXTS:
        continue
    # origen: cropped / train / val
    if "dataset_cropped_allclasses" in p.parts:
        source = "cropped"
    elif "train" in p.parts:
        source = "train"
    elif "val" in p.parts:
        source = "val"
    else:
        source = "otro"
    row = {
        "path": str(p),
        "source": source,
        "filename": p.name,
        "label": label_from_name(p.name),
        "md5": md5(p),
        "corrupt": False,
        "width": None,
        "height": None,
        "mode": None,
    }
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            row["width"], row["height"] = im.size
            row["mode"] = im.mode
    except Exception:
        row["corrupt"] = True
    rows.append(row)

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Total imagenes:", len(df))
print("\nPor origen y etiqueta:\n", df.groupby(["source", "label"]).size())
print("\nCorruptas:", int(df["corrupt"].sum()))
print("Duplicados exactos dentro de 'cropped':",
      int(df[df.source == "cropped"].duplicated("md5").sum()))

# Son train+val las mismas imagenes que cropped?
c = set(df[df.source == "cropped"]["md5"])
tv = set(df[df.source.isin(["train", "val"])]["md5"])
print("\nImagenes en train/val que tambien estan en cropped (por contenido):",
      len(c & tv), "de", len(tv))
print("\nResoluciones mas comunes:\n",
      df.groupby(["width", "height"]).size().sort_values(ascending=False).head(5))