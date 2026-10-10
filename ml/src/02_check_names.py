"""
Nombre: 02_check_names.py
Descripcion: Revisa los nombres repetidos entre train y val, y los nombres de
             cropped que no tienen foto original.
Fecha_creacion: 2026-10-10
Actualizacion: 2026-10-10
Version: 1.0
"""
import pandas as pd

df = pd.read_csv("ml/results/inventory.csv")
tv = df[df.source.isin(["train", "val"])]
cropped = df[df.source == "cropped"]

# 1) Nombres repetidos entre train y val
dup = tv[tv.duplicated("filename", keep=False)]
print("Archivos con nombre repetido:", len(dup))
print("\nPor origen y etiqueta:\n", dup.groupby(["source", "label"]).size())

# 2) Para esos nombres, el contenido es igual o distinto?
n_md5 = dup.groupby("filename")["md5"].nunique()
print("\nNombres repetidos con MISMO contenido (md5 igual):", int((n_md5 == 1).sum()))
print("Nombres repetidos con contenido DISTINTO:", int((n_md5 == 2).sum()))

# 3) Nombres de cropped sin foto original
sin_par = cropped[~cropped.filename.isin(set(tv.filename))]
print("\nCropped sin original:", len(sin_par))
print(sin_par.label.value_counts())
print("Ejemplos:", sin_par.filename.head(5).tolist())