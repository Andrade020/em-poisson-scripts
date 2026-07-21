# -*- coding: utf-8 -*-
"""
Validacao cruzada com pesos REPARAMETRIZADOS (coeficiente absoluto omega,
sem multiplicar por N^2) para o estimador penalizado do modelo 2.

Desenho, por dataset (NYC, Chicago, Seattle; janelas T=168, grade 10x10):
  - semanas ordenadas: 60% ajuste, 20% validacao (escolha de omega), 20% teste;
  - grid omega em {0, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5};
  - solver: EM penalizado (M-step L-BFGS-B), robusto ao peso;
  - metrica: MSE de previsao das contagens reportadas por (zona, hora):
      pred = lambda_hat * (1 - p_hat_t) * N_alvo;
  - transferencia: para cada dataset, avalia no teste o omega escolhido nos
    OUTROS datasets (razao vs o proprio otimo) — o teste da Remark de
    reparametrizacao do paper.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize
from pipeline import LOADERS, prepare

EPS = 1e-6
T = 168
GRID = 10
OMEGAS = [0.0, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5]


def build(df):
    d = df.copy()
    qlat = d["lat"].quantile([0.005, 0.995]); qlon = d["lon"].quantile([0.005, 0.995])
    ok = (~d["miss"]) & d["lat"].between(*qlat) & d["lon"].between(*qlon)
    zi = np.floor((d["lat"] - qlat.iloc[0]) / (qlat.iloc[1] - qlat.iloc[0] + 1e-9) * GRID).clip(0, GRID - 1)
    zj = np.floor((d["lon"] - qlon.iloc[0]) / (qlon.iloc[1] - qlon.iloc[0] + 1e-9) * GRID).clip(0, GRID - 1)
    d["zone"] = (zi * GRID + zj).where(ok)
    weeks = sorted(d["week"].unique())
    n = len(weeks)
    cut1, cut2 = int(0.6 * n), int(0.8 * n)
    parts = {}
    for nome, wk in [("fit", weeks[:cut1]), ("val", weeks[cut1:cut2]),
                     ("test", weeks[cut2:])]:
        dd = d[d["week"].isin(set(wk))]
        I = GRID * GRID
        m1 = np.zeros((I, T)); m0 = np.zeros(T)
        rep = dd[dd["zone"].notna()]
        np.add.at(m1, (rep["zone"].astype(int).values, rep["tow"].values), 1)
        mis = dd[dd["zone"].isna()]
        np.add.at(m0, mis["tow"].values, 1)
        parts[nome] = (m1, m0, len(wk))
    return parts


def make_pen(edges, groups, omega, shape):
    def pen_grad(lam):
        v, g = 0.0, np.zeros(shape)
        for G in groups:
            sub = lam[:, G]
            v += omega * (len(G) * (sub ** 2).sum() - (sub.sum(1) ** 2).sum())
            g[:, G] += 2 * omega * (len(G) * sub - sub.sum(1, keepdims=True))
        dif = lam[edges[:, 0]] - lam[edges[:, 1]]
        v += omega * (dif ** 2).sum()
        np.add.at(g, edges[:, 0], 2 * omega * dif)
        np.add.at(g, edges[:, 1], -2 * omega * dif)
        return v, g
    return pen_grad


def em_fit(M1, M0, N, omega, edges, groups, max_iter=120):
    I = M1.shape[0]
    pen_grad = make_pen(edges, groups, omega, M1.shape)

    def f(lam):
        S = lam.sum(0)
        return float((N * S - M0 * np.log(S)).sum() - (M1 * np.log(lam)).sum()
                     + pen_grad(lam)[0])

    lam = np.full((I, T), max(M1.sum() / (N * I * T), 0.05))
    fv, stall = f(lam), 0
    for _ in range(max_iter):
        S = lam.sum(0)
        C = M1 + M0[None, :] * lam / S[None, :]

        def q(x):
            l = x.reshape(I, T)
            return (N * l.sum(0)).sum() - (C * np.log(l)).sum() + pen_grad(l)[0]

        def gq(x):
            l = x.reshape(I, T)
            return (N - C / l + pen_grad(l)[1]).ravel()

        res = minimize(q, lam.ravel(), jac=gq, method="L-BFGS-B",
                       bounds=[(EPS, None)] * (I * T),
                       options=dict(maxiter=150, ftol=1e-12))
        lam = res.x.reshape(I, T)
        fn = f(lam)
        rel = (fv - fn) / max(abs(fv), 1.0)
        fv = fn
        stall = stall + 1 if rel < 1e-10 else 0
        if stall >= 5:
            break
    return lam


def mse_pred(lam, p_t, M1_alvo, N_alvo):
    pred = lam * (1 - p_t)[None, :] * N_alvo
    return float(((pred - M1_alvo) ** 2).mean())


if __name__ == "__main__":
    edges = []
    for r in range(GRID):
        for c in range(GRID):
            k = r * GRID + c
            if r + 1 < GRID:
                edges.append((k, k + GRID))
            if c + 1 < GRID:
                edges.append((k, k + 1))
    edges = np.array(edges)
    groups = [[d_ * 24 + h for d_ in range(5)] for h in range(24)] + \
             [[d_ * 24 + h for d_ in (5, 6)] for h in range(24)]

    out = {}
    for name in ["nyc", "chicago", "seattle"]:
        t0 = time.perf_counter()
        df = prepare(LOADERS[name]())
        parts = build(df)
        M1f, M0f, Nf = parts["fit"]
        M1v, M0v, Nv = parts["val"]
        M1t, M0t, Nt = parts["test"]
        p_t = M0f / np.maximum(M0f + M1f.sum(0), 1)
        curva_val, curva_test = {}, {}
        for om in OMEGAS:
            lam = em_fit(M1f, M0f, Nf, om, edges, groups)
            curva_val[om] = round(mse_pred(lam, p_t, M1v, Nv), 3)
            curva_test[om] = round(mse_pred(lam, p_t, M1t, Nt), 3)
        om_star = min(curva_val, key=curva_val.get)
        out[name] = dict(
            semanas=dict(fit=Nf, val=Nv, test=Nt),
            curva_validacao={f"{om:g}": v for om, v in curva_val.items()},
            curva_teste={f"{om:g}": v for om, v in curva_test.items()},
            omega_escolhido=f"{om_star:g}",
            mse_teste_omega_escolhido=curva_test[om_star],
            mse_teste_sem_pen=curva_test[0.0],
            tempo_s=round(time.perf_counter() - t0, 1))
        print(name, "->", json.dumps(out[name]["curva_validacao"]),
              "omega*:", out[name]["omega_escolhido"], flush=True)

    # transferencia: omega escolhido em A avaliado no teste de B
    transf = {}
    for a in out:
        for b in out:
            if a == b:
                continue
            om_a = out[a]["omega_escolhido"]
            mse_b = out[b]["curva_teste"][om_a]
            best_b = min(out[b]["curva_teste"].values())
            transf[f"{a}->{b}"] = dict(
                mse=mse_b, razao_vs_otimo=round(mse_b / best_b, 3))
    out["transferencia"] = transf
    print(json.dumps(out, indent=1, ensure_ascii=False))
    json.dump(out, open("resultados_cv_repar.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
