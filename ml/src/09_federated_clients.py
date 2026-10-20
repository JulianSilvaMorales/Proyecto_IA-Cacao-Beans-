"""
Nombre: 09_federated_clients.py
Descripcion: Reparte data/splits/train.csv entre N clientes simulados para
             Federated Learning (reparto estratificado por clase, sin repetir
             granos). Guarda un CSV por cliente en data/splits/federated/ y
             los conteos en ml/results/federated_counts.csv. val y test no se
             reparten: son globales.
Fecha_creacion: 2026-10-19
Actualizacion: 2026-10-19
Autor: Julian Andres Silvas Morales
Version: 1.0
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_CLIENTS = 5
OUT = Path("data/splits/federated")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv("data/splits/train.csv").reset_index(drop=True)

skf = StratifiedKFold(n_splits=N_CLIENTS, shuffle=True, random_state=SEED)
clients = []
for i, (_, idx) in enumerate(skf.split(train, train["label"])):
    part = train.iloc[idx].sort_values("path")
    part.to_csv(OUT / f"client_{i}.csv", index=False)
    clients.append(part)

rows = []
for i, c in enumerate(clients):
    vc = c["label"].value_counts()
    rows.append({"cliente": i, "total": len(c),
                 "sana": int(vc.get("sana", 0)), "mala": int(vc.get("mala", 0))})
counts = pd.DataFrame(rows)
counts["pct_mala"] = (counts["mala"] / counts["total"] * 100).round(1)
counts.to_csv("ml/results/federated_counts.csv", index=False)
print(counts.to_string(index=False))

# Verificacion: sin repetidos entre clientes y la union es todo train
paths = [set(c["path"]) for c in clients]
for a in range(N_CLIENTS):
    for b in range(a + 1, N_CLIENTS):
        assert not paths[a] & paths[b], f"Granos repetidos entre clientes {a} y {b}"
assert set().union(*paths) == set(train["path"]), "Faltan granos de train"
assert sum(len(c) for c in clients) == len(train)
print(f"\nOK: {N_CLIENTS} clientes, sin solapamiento, suman {len(train)}")
