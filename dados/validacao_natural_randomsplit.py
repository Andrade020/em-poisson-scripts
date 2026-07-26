# -*- coding: utf-8 -*-
"""
Random event-split da validacao natural (coluna "random" da Tabela 5).

Desenho honesto contra overfitting: TODOS os eventos (reportados e
faltantes) sao divididos ao meio; a metade A fornece as odds MNAR
p_i/(1-p_i) por zona; a metade B fornece a alocacao do modelo
(M1, M0_t), a verdade (faltantes rotulados de B) e as TVs. Nada da
metade B entra na estimativa do ajuste.

Para eliminar a sensibilidade ao seed, repetimos R=50 splits
independentes e reportamos media e desvio-padrao das TVs.
Saida: resultados_validacao_natural_randomsplit.json.
"""
import json
import numpy as np
from pipeline import load_chicago, load_nyc, prepare

T = 168
R = 50

def tvd(E, q):
    return 0.5 * float(np.abs(E / E.sum() - q).sum())

def one_split(df, zones, zidx, rng):
    I = len(zones)
    inA = rng.random(len(df)) < 0.5
    A, B = df[inA], df[~inA]

    def counts(part):
        rep = part[(~part["miss"]) & part["zone_known"].notna()]
        mis = part[part["miss"]]
        mis_lab = mis[mis["zone_known"].notna()]
        M1 = np.zeros((I, T))
        np.add.at(M1, (rep["zone_known"].map(zidx).values, rep["tow"].values), 1)
        M0_t = np.zeros(T)
        np.add.at(M0_t, mis["tow"].values, 1)
        mis_i = np.zeros(I)
        np.add.at(mis_i, mis_lab["zone_known"].map(zidx).values, 1)
        return M1, M0_t, mis_i

    M1A, _, misA = counts(A)
    M1B, M0B, misB = counts(B)
    p_i = np.clip(misA / np.maximum(misA + M1A.sum(1), 1), 1e-9, 1 - 1e-9)
    colsum = np.maximum(M1B.sum(0), 1)
    E_model = (M1B / colsum[None, :] * M0B[None, :]).sum(1)
    E_unif = np.full(I, M0B.sum() / I)
    E_adj = E_model * (p_i / (1 - p_i))
    q_act = misB / misB.sum()
    return tvd(E_model, q_act), tvd(E_adj, q_act), tvd(E_unif, q_act)

def run(df, name):
    zones = sorted(df.loc[df["zone_known"].notna(), "zone_known"].unique())
    zidx = {z: k for k, z in enumerate(zones)}
    rng = np.random.default_rng(0)
    res = np.array([one_split(df, zones, zidx, rng) for _ in range(R)])
    keys = ["TV_modelo", "TV_ajustado_random_split", "TV_uniforme"]
    out = dict(dataset=name, n_splits=R)
    for j, k in enumerate(keys):
        out[k] = round(float(res[:, j].mean()), 4)
        out[k + "_sd"] = round(float(res[:, j].std()), 4)
    return out

if __name__ == "__main__":
    results = []
    results.append(run(prepare(load_chicago()), "chicago_CA"))
    dfn = prepare(load_nyc())
    dfn = dfn[(dfn.ts >= "2013-01-01") & (dfn.ts <= "2019-12-31")]
    results.append(run(dfn, "nyc_boroughs"))
    for r in results:
        print(json.dumps(r, indent=1), flush=True)
    json.dump(results, open("resultados_validacao_natural_randomsplit.json",
              "w"), indent=1)
