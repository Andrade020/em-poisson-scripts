# -*- coding: utf-8 -*-
"""
DATATRAN (PRF, acidentes em rodovias federais) 2017-2025:
 1. faltante NATURAL de coordenadas por ano (e teste MNAR por UF quando houver);
 2. benchmark SEMI-SINTETICO brasileiro (verdade completa + mecanismo do artigo):
    (a) Brasil, zonas = 27 UFs;  (b) RJ, zonas = grade 6x6 sobre o estado.
Metricas como em semisintetico.py: reducao de erro do corrigido vs descartar e
cobertura do IC 90% de p_t (R replicas de mascaramento binomial).
"""
import glob
import io
import json
import zipfile
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
T = 168
R = 200
Z = 1.6449

frames = []
for f in sorted(glob.glob("datatran_20*_ok.zip")) + ["datatran_1-WO3SfNrwwZ5_l7fRTiwBKRw7mi1-HUq.zip"]:
    z = zipfile.ZipFile(f)
    nm = [n for n in z.namelist() if n.startswith("datatran")][0]
    ano = int(nm[8:12])
    if ano < 2017:
        continue
    df = pd.read_csv(io.BytesIO(z.read(nm)), sep=";", encoding="latin-1",
                     decimal=",", low_memory=False)
    df["ano"] = ano
    frames.append(df[["ano", "data_inversa", "horario", "uf", "municipio",
                      "latitude", "longitude", "classificacao_acidente"]])
df = pd.concat(frames, ignore_index=True).drop_duplicates()
df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
bad = df["lat"].abs() < 1e-9
df.loc[bad, ["lat", "lon"]] = np.nan
df["miss"] = df["lat"].isna() | df["lon"].isna()
ts = pd.to_datetime(df["data_inversa"].astype(str) + " " + df["horario"].astype(str),
                    errors="coerce", format="mixed")
df["tow"] = ts.dt.dayofweek * 24 + ts.dt.hour
df = df.dropna(subset=["tow"])
df["tow"] = df["tow"].astype(int)
Nsem = ts.dt.date.nunique() / 7

out = dict(anos=sorted(df["ano"].unique().tolist()), n_eventos=int(len(df)),
           semanas=round(Nsem, 1))
out["faltante_por_ano_%"] = {int(a): round(100 * float(g["miss"].mean()), 2)
                             for a, g in df.groupby("ano")}
out["faltante_global_%"] = round(100 * float(df["miss"].mean()), 3)

# MNAR por UF (se houver faltante relevante)
if df["miss"].mean() > 0.001:
    g = df.groupby("uf")["miss"].agg(["sum", "count"])
    g = g[g["count"] > 500]
    p = g["sum"] / g["count"]
    out["faltante_por_UF"] = dict(min=round(float(p.min()), 4),
                                  max=round(float(p.max()), 4))

def semisint(sub, zones, label):
    M = np.zeros((len(zones), T))
    zmap = {z: k for k, z in enumerate(zones)}
    zz = sub["zona"].map(zmap)
    ok = zz.notna()
    np.add.at(M, (zz[ok].astype(int).values, sub.loc[ok, "tow"].values), 1)
    N = Nsem
    lam_true = M / N
    hours = np.arange(T) % 24
    p_t = 0.15 + 0.20 * (np.cos(2 * np.pi * (hours - 3) / 24) * 0.5 + 0.5)
    ec, eu, cov = [], [], []
    for _ in range(R):
        M1 = rng.binomial(M.astype(int), (1 - p_t)[None, :])
        M0 = (M - M1).sum(0)
        S_hat = (M1.sum(0) + M0) / N
        lam_c = S_hat * M1 / np.maximum(M1.sum(0), 1)
        lam_u = M1 / N
        ec.append(np.abs(lam_c - lam_true).sum() / lam_true.sum())
        eu.append(np.abs(lam_u - lam_true).sum() / lam_true.sum())
        tot = M0 + M1.sum(0)
        p_hat = M0 / np.maximum(tot, 1)
        se = np.sqrt(p_hat * (1 - p_hat) / np.maximum(tot, 1))
        cov.append(float(((p_t >= p_hat - Z * se) & (p_t <= p_hat + Z * se)).mean()))
    return {f"{label}": dict(
        zonas=len(zones), eventos=int(M.sum()),
        erro_rel_corrigido=round(float(np.mean(ec)), 4),
        erro_rel_descartando=round(float(np.mean(eu)), 4),
        reducao_erro_pct=round(100 * (1 - np.mean(ec) / np.mean(eu)), 1),
        cobertura_IC90_p=round(float(np.mean(cov)), 3))}

# (a) Brasil, zonas = UF
comp = df[~df["miss"]].copy()
comp["zona"] = comp["uf"]
ufs = sorted(comp["uf"].dropna().unique().tolist())
out["semisintetico"] = semisint(comp, ufs, "Brasil_UF")

# (b) RJ, grade 6x6
rj = comp[comp["uf"] == "RJ"].copy()
qlat = rj["lat"].quantile([0.01, 0.99]); qlon = rj["lon"].quantile([0.01, 0.99])
rj = rj[rj["lat"].between(*qlat) & rj["lon"].between(*qlon)]
zi = np.floor((rj["lat"] - qlat.iloc[0]) / (qlat.iloc[1] - qlat.iloc[0] + 1e-9) * 6).clip(0, 5)
zj = np.floor((rj["lon"] - qlon.iloc[0]) / (qlon.iloc[1] - qlon.iloc[0] + 1e-9) * 6).clip(0, 5)
rj["zona"] = (zi * 6 + zj).astype(int)
out["semisintetico"].update(semisint(rj, list(range(36)), "RJ_grade6x6"))

print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("resultados_datatran.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
