# -*- coding: utf-8 -*-
"""
OOS do Rio com folds ALEATORIOS de semanas (corrige a rodada com teste no
ultimo bloco, que tem exposicao degenerada — ver RELATORIO_EMPIRICO §5).
Persistido como fonte primaria da linha "Rio +28%" da Tabela dose-resposta.
5 folds, semente fixa; MSE de previsao da intensidade TOTAL por janela.
"""
import json
import numpy as np
from rio_paper_data import load

rng = np.random.default_rng(1)
M1, M0, N = load()
P = 3
perm = rng.permutation(N)
folds = np.array_split(perm, 5)

por_prio = {p: dict(corr=[], unc=[]) for p in range(P)}
pool_corr, pool_unc = [], []
for f in folds:
    te = np.zeros(N, bool)
    te[f] = True
    Ntr, Nte = int((~te).sum()), int(te.sum())
    for p in range(P):
        M1tr, M0tr = M1[p][:, :, ~te].sum(2), M0[p][:, ~te].sum(1)
        M1te, M0te = M1[p][:, :, te].sum(2), M0[p][:, te].sum(1)
        tot_te = M1te.sum(0) + M0te
        pred_c = (M1tr.sum(0) + M0tr) / Ntr * Nte
        pred_u = M1tr.sum(0) / Ntr * Nte
        mse_c = float(((pred_c - tot_te) ** 2).mean())
        mse_u = float(((pred_u - tot_te) ** 2).mean())
        por_prio[p]["corr"].append(mse_c)
        por_prio[p]["unc"].append(mse_u)
        pool_corr.append(mse_c)
        pool_unc.append(mse_u)

out = dict(
    desenho="5 folds aleatorios de semanas, semente 1 (exposicao por semana "
            "inteira; evita a cauda degenerada do arquivo de replicacao)",
    MSE_corrigido=round(float(np.mean(pool_corr)), 2),
    MSE_descartando=round(float(np.mean(pool_unc)), 2),
    razao_desc_sobre_corr=round(float(np.mean(pool_unc) / np.mean(pool_corr)), 3),
    ganho_pct=round(100 * (1 - np.mean(pool_corr) / np.mean(pool_unc)), 1),
    por_prioridade={
        f"prio{p}": dict(
            MSE_corrigido=round(float(np.mean(v["corr"])), 2),
            MSE_descartando=round(float(np.mean(v["unc"])), 2),
            ganho_pct=round(100 * (1 - np.mean(v["corr"]) / np.mean(v["unc"])), 1))
        for p, v in por_prio.items()})
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("resultados_rio_oos_random.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
