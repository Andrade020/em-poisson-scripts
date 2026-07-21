# -*- coding: utf-8 -*-
"""
Resposta ao referee (M6): valor estatistico da penalizacao no regime esparso.
Amputation no grid 6x6 do estado do RJ (DATATRAN 2017-25), mecanismo do
artigo (p_t medio 0.25, independente de zona), e ajuste do estimador
PENALIZADO com pesos reparametrizados (coeficiente absoluto omega, sem N^2).
Para cada replica: omega escolhido por validacao interna (split de semanas
via thinning binomial das contagens mascaradas), erro final medido contra a
verdade completa. R=20 replicas; grid omega em {0, 3e2, 1e3, 3e3, 1e4, 3e4}.
"""
import glob
import io
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.optimize import minimize

rng = np.random.default_rng(31)
T = 168
G6 = 6
I = G6 * G6
EPS = 1e-6
R = 20
OMEGAS = [0.0, 3e2, 1e3, 3e3, 1e4, 3e4]

# ---------------- dados RJ (reuso da logica de datatran_analise) ------------
frames = []
for f in sorted(glob.glob("datatran_20*_ok.zip")) + \
        ["datatran_1-WO3SfNrwwZ5_l7fRTiwBKRw7mi1-HUq.zip"]:
    z = zipfile.ZipFile(f)
    nm = [n for n in z.namelist() if n.startswith("datatran")][0]
    if int(nm[8:12]) < 2017:
        continue
    df = pd.read_csv(io.BytesIO(z.read(nm)), sep=";", encoding="latin-1",
                     decimal=",", low_memory=False)
    frames.append(df[["data_inversa", "horario", "uf", "latitude", "longitude"]])
df = pd.concat(frames, ignore_index=True).drop_duplicates()
df = df[df["uf"] == "RJ"].copy()
df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["lat", "lon"])
ts = pd.to_datetime(df["data_inversa"].astype(str) + " " + df["horario"].astype(str),
                    errors="coerce", format="mixed")
df["tow"] = (ts.dt.dayofweek * 24 + ts.dt.hour)
df = df.dropna(subset=["tow"])
qlat = df["lat"].quantile([0.01, 0.99]); qlon = df["lon"].quantile([0.01, 0.99])
df = df[df["lat"].between(*qlat) & df["lon"].between(*qlon)]
zi = np.floor((df["lat"] - qlat.iloc[0]) / (qlat.iloc[1] - qlat.iloc[0] + 1e-9) * G6).clip(0, G6 - 1)
zj = np.floor((df["lon"] - qlon.iloc[0]) / (qlon.iloc[1] - qlon.iloc[0] + 1e-9) * G6).clip(0, G6 - 1)
zone = (zi * G6 + zj).astype(int)
M = np.zeros((I, T))
np.add.at(M, (zone.values, df["tow"].astype(int).values), 1)
N = ts.dt.date.nunique() / 7
lam_true = M / N

hours = np.arange(T) % 24
p_t = 0.15 + 0.20 * (np.cos(2 * np.pi * (hours - 3) / 24) * 0.5 + 0.5)

# ------------------------- penalizacao e solver EM --------------------------
edges = []
for r_ in range(G6):
    for c_ in range(G6):
        k = r_ * G6 + c_
        if r_ + 1 < G6:
            edges.append((k, k + G6))
        if c_ + 1 < G6:
            edges.append((k, k + 1))
edges = np.array(edges)
groups = [[d_ * 24 + h for d_ in range(5)] for h in range(24)] + \
         [[d_ * 24 + h for d_ in (5, 6)] for h in range(24)]


def pen_grad(lam, om):
    v, g = 0.0, np.zeros_like(lam)
    for Gr in groups:
        sub = lam[:, Gr]
        v += om * (len(Gr) * (sub ** 2).sum() - (sub.sum(1) ** 2).sum())
        g[:, Gr] += 2 * om * (len(Gr) * sub - sub.sum(1, keepdims=True))
    dif = lam[edges[:, 0]] - lam[edges[:, 1]]
    v += om * (dif ** 2).sum()
    np.add.at(g, edges[:, 0], 2 * om * dif)
    np.add.at(g, edges[:, 1], -2 * om * dif)
    return v, g


def em_fit(M1, M0, Neff, om, max_iter=80):
    def f(lam):
        S = lam.sum(0)
        return float((Neff * S - M0 * np.log(np.maximum(S, 1e-12))).sum()
                     - (M1 * np.log(lam)).sum() + pen_grad(lam, om)[0])

    lam = np.full((I, T), max(M1.sum() / (Neff * I * T), 0.01))
    fv, stall = f(lam), 0
    for _ in range(max_iter):
        S = lam.sum(0)
        C = M1 + M0[None, :] * lam / np.maximum(S, 1e-12)[None, :]

        def q(x):
            l = x.reshape(I, T)
            return (Neff * l.sum(0)).sum() - (C * np.log(l)).sum() + pen_grad(l, om)[0]

        def gq(x):
            l = x.reshape(I, T)
            return (Neff - C / l + pen_grad(l, om)[1]).ravel()

        res = minimize(q, lam.ravel(), jac=gq, method="L-BFGS-B",
                       bounds=[(EPS, None)] * (I * T),
                       options=dict(maxiter=120, ftol=1e-12))
        lam = res.x.reshape(I, T)
        fn = f(lam)
        rel = (fv - fn) / max(abs(fv), 1.0)
        fv = fn
        stall = stall + 1 if rel < 1e-9 else 0
        if stall >= 4:
            break
    return lam


def rel_err(lam):
    return float(np.abs(lam - lam_true).sum() / lam_true.sum())


# ------------------------------- replicas -----------------------------------
res = {f"{om:g}": [] for om in OMEGAS}
res_cv, res_unpen_closed = [], []
for rep in range(R):
    M1 = rng.binomial(M.astype(int), (1 - p_t)[None, :])
    M0 = (M - M1).sum(0)
    # selecao interna de omega: thinning binomial 50/50 das contagens
    M1a = rng.binomial(M1.astype(int), 0.5)
    M1b = M1 - M1a
    M0a = rng.binomial(M0.astype(int), 0.5)
    M0b = M0 - M0a
    p_hat = M0b / np.maximum(M0b + M1b.sum(0), 1)
    val_scores = {}
    for om in OMEGAS:
        lam_a = em_fit(M1a, M0a, N / 2, om)
        pred_b = lam_a * (1 - p_hat)[None, :] * (N / 2)
        val_scores[om] = float(((pred_b - M1b) ** 2).mean())
    om_star = min(val_scores, key=val_scores.get)
    # ajuste final com todos os dados mascarados
    for om in OMEGAS:
        lam = em_fit(M1, M0, N, om)
        res[f"{om:g}"].append(rel_err(lam))
        if om == om_star:
            res_cv.append(rel_err(lam))
    # forma fechada nao penalizada (referencia da Tabela de amputacao)
    S_hat = (M1.sum(0) + M0) / N
    lam_cf = S_hat * M1 / np.maximum(M1.sum(0), 1)
    res_unpen_closed.append(rel_err(lam_cf))
    print(f"replica {rep+1}/{R}: omega*={om_star:g}", flush=True)

out = dict(
    dataset="DATATRAN RJ 6x6", replicas=R, semanas=round(N, 1),
    eventos=int(M.sum()), p_medio=round(float(p_t.mean()), 3),
    erro_por_omega={k: dict(media=round(float(np.mean(v)), 4),
                            dp=round(float(np.std(v)), 4))
                    for k, v in res.items()},
    erro_omega_cv=dict(media=round(float(np.mean(res_cv)), 4),
                       dp=round(float(np.std(res_cv)), 4)),
    erro_fechado_sem_pen=dict(media=round(float(np.mean(res_unpen_closed)), 4),
                              dp=round(float(np.std(res_unpen_closed)), 4)))
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("resultados_amputacao_rj_pen.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
