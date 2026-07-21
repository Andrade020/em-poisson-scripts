# -*- coding: utf-8 -*-
"""
Benchmark SEMI-SINTETICO: condicoes do artigo satisfeitas por construcao.

Ideia: Chicago 2019+ e Seattle 2018+ sao >98,5% completos. Tratamos os eventos
com localizacao como a "verdade" (processo completo real, com toda a nao-
homogeneidade espaco-temporal de dados reais), e IMPOMOS o mecanismo de
faltante do artigo: cada evento perde a localizacao com probabilidade p_{c,t}
que depende da hora-da-semana e do tipo, MAS NAO da zona (MAR espacial).
Entao:
  - a verdade lambda(i,t) e conhecida (contagens completas);
  - as hipoteses do modelo valem por construcao;
  - medimos a recuperacao do estimador corrigido (eq. 6) vs descartar,
    a cobertura do IC de Fisher para p_t, e
  - num segundo cenario VIOLAMOS a hipotese (p depende da zona com dispersao
    delta) e comparamos o vies observado por zona com o previsto pela teoria:
        lambda_hat_i / lambda_i  ->  (1-p_i)/(1-p_bar_t)   [previsao]
Replicamos o mascaramento R vezes (thinning binomial das contagens).
"""
import json
import numpy as np
import pandas as pd
from pipeline import load_chicago, load_seattle, prepare

rng = np.random.default_rng(2026)
T = 168
GRID = 6            # 36 zonas (mantem contagens densas por celula)
R = 200             # replicas de mascaramento
Z = 1.6449          # quantil 95% -> IC de 90%

def build_counts(df):
    """Retorna M[i,t] contagens completas e N (semanas via dias/7)."""
    d = df[~df["miss"]].copy()
    qlat = d["lat"].quantile([0.005, 0.995]); qlon = d["lon"].quantile([0.005, 0.995])
    d = d[d["lat"].between(*qlat) & d["lon"].between(*qlon)]
    zi = np.floor((d["lat"] - qlat.iloc[0]) / (qlat.iloc[1] - qlat.iloc[0] + 1e-9) * GRID).clip(0, GRID - 1)
    zj = np.floor((d["lon"] - qlon.iloc[0]) / (qlon.iloc[1] - qlon.iloc[0] + 1e-9) * GRID).clip(0, GRID - 1)
    zone = (zi * GRID + zj).astype(int)
    M = np.zeros((GRID * GRID, T))
    np.add.at(M, (zone.values, d["tow"].values), 1)
    N = d["ts"].dt.date.nunique() / 7
    return M, N

def run(name, loader, window):
    df = prepare(loader())
    df = df[(df.ts >= window[0]) & (df.ts <= window[1])]
    M, N = build_counts(df)          # (I,T) verdade completa
    I = M.shape[0]
    tot_t = M.sum(0)                  # total verdadeiro por janela
    lam_true = M / N                  # intensidade "verdadeira" (eventos/semana)

    # p_t realista: mais faltante de madrugada, media ~0.25 (nivel do Rio)
    hours = np.arange(T) % 24
    p_t = 0.15 + 0.20 * (np.cos(2 * np.pi * (hours - 3) / 24) * 0.5 + 0.5)  # 0.15-0.35

    out = dict(dataset=name, I=I, T=T, semanas=round(N, 1),
               eventos=int(M.sum()), p_medio=round(float(p_t.mean()), 3))

    # ---------------- Cenario A: hipoteses valem (p nao depende de zona) ----
    errA_corr, errA_unc, cover_p = [], [], []
    for _ in range(R):
        M1 = rng.binomial(M.astype(int), (1 - p_t)[None, :])
        M0 = (M - M1).sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            S_hat = (M1.sum(0) + M0) / N
            lam_c = np.where(M1.sum(0) > 0, S_hat * M1 / np.maximum(M1.sum(0), 1), 0.0)
        lam_u = M1 / N
        errA_corr.append(np.abs(lam_c - lam_true).sum() / lam_true.sum())
        errA_unc.append(np.abs(lam_u - lam_true).sum() / lam_true.sum())
        # IC de Fisher para p_t: Var = p(1-p)/(N D S) com D S = total esperado/sem
        p_hat = M0 / np.maximum(M0 + M1.sum(0), 1)
        se = np.sqrt(p_hat * (1 - p_hat) / np.maximum(M0 + M1.sum(0), 1))
        cover_p.append(float(((p_t >= p_hat - Z * se) & (p_t <= p_hat + Z * se)).mean()))
    out["A_hipoteses_valem"] = dict(
        erro_rel_corrigido=round(float(np.mean(errA_corr)), 4),
        dp_corrigido=round(float(np.std(errA_corr)), 4),
        erro_rel_descartando=round(float(np.mean(errA_unc)), 4),
        dp_descartando=round(float(np.std(errA_unc)), 4),
        reducao_erro_pct=round(100 * (1 - np.mean(errA_corr) / np.mean(errA_unc)), 1),
        cobertura_IC90_p=round(float(np.mean(cover_p)), 3),
        se_mc_cobertura=round(float(np.std(cover_p) / np.sqrt(len(cover_p))), 4))

    # ---------------- Cenario B: violacao controlada (p depende da zona) ----
    # p_{i,t} = p_t * fator_i, fator_i em [1-delta, 1+delta] (dispersao entre zonas)
    res_B = {}
    for delta in (0.2, 0.5, 0.9):
        fac = 1 + delta * (2 * rng.random(I) - 1)
        P = np.clip(p_t[None, :] * fac[:, None], 0, 0.95)
        ratio_obs = np.zeros(I)
        for _ in range(50):
            M1 = rng.binomial(M.astype(int), 1 - P)
            M0 = (M - M1).sum(0)
            S_hat = (M1.sum(0) + M0) / N
            lam_c = S_hat * M1 / np.maximum(M1.sum(0), 1)
            ratio_obs += (lam_c.sum(1) / np.maximum(lam_true.sum(1), 1e-9)) / 50
        # previsao teorica (agregada em t com pesos lambda):
        pbar_t = (P * M).sum(0) / np.maximum(M.sum(0), 1)
        ratio_pred = ((1 - P) * M).sum(1) / np.maximum(((1 - pbar_t)[None, :] * M).sum(1), 1e-9)
        corr = float(np.corrcoef(ratio_obs, ratio_pred)[0, 1])
        res_B[f"delta={delta}"] = dict(
            vies_zona_min=round(float(ratio_obs.min() - 1), 3),
            vies_zona_max=round(float(ratio_obs.max() - 1), 3),
            correlacao_obs_vs_teoria=round(corr, 3),
            erro_max_da_previsao=round(float(np.abs(ratio_obs - ratio_pred).max()), 4))
    out["B_violacao_MNAR"] = res_B
    return out

if __name__ == "__main__":
    results = []
    for name, loader, win in [
        ("chicago", load_chicago, ("2019-01-01", "2026-07-01")),
        ("seattle", load_seattle, ("2018-01-01", "2026-07-01")),
    ]:
        r = run(name, loader, win)
        results.append(r)
        print(json.dumps(r, indent=1, ensure_ascii=False))
    json.dump(results, open("resultados_semisintetico.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
