"""
Nombre: 01_check_pairs.py
Descripcion: Verifica si los archivos de cropped y de train/val corresponden
             a los mismos granos (comparando nombres de archivo).
Fecha_creacion: 2026-10-09
Actualizacion: 2026-10-09
Autor: Julian Andres Silvas Morales
Version: 1.0
"""
import pandas as pd

df = pd.read_csv("ml/results/inventory.csv")
cropped = df[df.source == "cropped"]
tv = df[df.source.isin(["train", "val"])]

names_c = set(cropped.filename)
names_tv = set(tv.filename)
print("Nombres en cropped:", len(names_c))
print("Nombres en train/val:", len(names_tv))
print("Nombres en comun:", len(names_c & names_tv))

print("\nTamano de imagen por origen (mediana):")
print(df.groupby("source")[["width", "height"]].median())

print("\nEjemplos de rutas:")
for n in ["mala (1).jpg", "sana (5).jpg"]:
    print(df[df.filename == n][["source", "path", "width", "height"]].to_string(index=False))