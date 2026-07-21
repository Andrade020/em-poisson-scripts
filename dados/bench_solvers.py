# -*- coding: utf-8 -*-
"""
Benchmark justo de solvers (resposta ao referee M2-M4): PG-Armijo SEM teto,
FISTA projetado (com backtracking e restart), L-BFGS-B direto em F_N, e EM
penalizado. Dados reais (NYC/Chicago/Seattle, grade 10x10, T=168, w=0).
Timing: mediana de 3 repeticoes (ajustes deterministicos; semente de dados
fixa no pipeline). Metrica de convergencia: iteracoes/tempo ate gap relativo
1e-8 vs o melhor objetivo encontrado por qualquer solver.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize
from pipeline import LOADERS, prepare

EPS = 1e-6
T = 168
GRID = 10
REPS = 3
CAP_FO = 20000       # teto alto para metodos de 1a ordem (reportado se atingido)


def build_counts(name):
    df = prepare(LOADERS[name]())
    d = df.copy()
    qlat = d["lat"].quantile([0.005, 0.995]); qlon = d["lon"].quantile([0.005, 0.995])
    ok = (~d["miss"]) & d["lat"].between(*qlat) & d["lon"].between(*qlon)
    zi = np.floor((d["lat"] - qlat.iloc[0]) / (qlat.iloc[1] - qlat.iloc[0] + 1e-9) * GRID).clip(0, GRID - 1)
    zj = np.floor((d["lon"] - qlon.iloc[0]) / (qlon.iloc[1] - qlon.iloc[0] + 1e-9) * GRID).clip(0, GRID - 1)
    d["zone"] = (zi * GRID + zj).where(ok)
    I = GRID * GRID
    M1 = np.zeros((I, T)); M0 = np.zeros(T)
    rep = d[d["zone"].notna()]
    np.add.at(M1, (rep["zone"].astype(int).values, rep["tow"].values), 1)
    mis = d[d["zone"].isna()]
    np.add.at(M0, mis["tow"].values, 1)
    N = d["ts"].dt.date.nunique() / 7
    return M1, M0, N


def make_f(M1, M0, N):
    def f(lam):
        S = lam.sum(0)
        return float((N * S - M0 * np.log(S)).sum() - (M1 * np.log(lam)).sum())

    def gr(lam):
        S = lam.sum(0)
        return (N - M0 / S)[None, :] - M1 / lam
    return f, gr


def run_pg(f, gr, lam0, cap=CAP_FO):
    lam, fv = lam0.copy(), None
    fv = f(lam)
    hist = [(0, 0.0, fv)]
    t0 = time.perf_counter()
    step, stall = 1.0, 0
    for it in range(1, cap + 1):
        g = gr(lam)
        t = step
        for _ in range(60):
            ln = np.maximum(lam - t * g, EPS)
            fn = f(ln)
            if fn <= fv + 1e-4 * (g * (ln - lam)).sum():
                break
            t *= 0.5
        rel = (fv - fn) / max(abs(fv), 1.0)
        lam, fv, step = ln, fn, min(t / 0.5, 100.0)
        hist.append((it, time.perf_counter() - t0, fv))
        stall = stall + 1 if rel < 1e-12 else 0
        if stall >= 8:
            break
    return lam, hist


def run_fista(f, gr, lam0, cap=CAP_FO):
    """FISTA projetado com backtracking e restart por monotonia."""
    lam = lam0.copy(); y = lam0.copy()
    L = 1.0; tk = 1.0
    fv = f(lam)
    hist = [(0, 0.0, fv)]
    t0 = time.perf_counter()
    stall = 0
    for it in range(1, cap + 1):
        g = gr(y)
        fy = f(y)
        while True:
            ln = np.maximum(y - g / L, EPS)
            d = ln - y
            if f(ln) <= fy + (g * d).sum() + 0.5 * L * (d * d).sum() or L > 1e12:
                break
            L *= 2.0
        tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        fn = f(ln)
        if fn > fv:                      # restart
            y, tk = lam.copy(), 1.0
            L = max(L / 2, 1e-8)
            fn2 = fv
        else:
            y = ln + ((tk - 1) / tk1) * (ln - lam)
            lam, fn2 = ln, fn
        rel = (fv - fn2) / max(abs(fv), 1.0)
        fv = fn2
        tk = tk1
        L = max(L / 2, 1e-8)
        hist.append((it, time.perf_counter() - t0, fv))
        stall = stall + 1 if rel < 1e-12 else 0
        if stall >= 8:
            break
    return lam, hist


def run_lbfgs(f, gr, lam0, shape):
    t0 = time.perf_counter()
    res = minimize(lambda x: f(x.reshape(shape)), lam0.ravel(),
                   jac=lambda x: gr(x.reshape(shape)).ravel(),
                   method="L-BFGS-B", bounds=[(EPS, None)] * lam0.size,
                   options=dict(maxiter=CAP_FO, ftol=1e-14, gtol=1e-10))
    return res.x.reshape(shape), [(res.nit, time.perf_counter() - t0,
                                   float(res.fun))]


def run_em(f, M1, M0, N, lam0, cap=200):
    I = M1.shape[0]
    lam, fv = lam0.copy(), f(lam0)
    hist = [(0, 0.0, fv)]
    t0 = time.perf_counter()
    stall = 0
    for it in range(1, cap + 1):
        S = lam.sum(0)
        C = M1 + M0[None, :] * lam / S[None, :]

        def q(x):
            l = x.reshape(I, T)
            return (N * l.sum(0)).sum() - (C * np.log(l)).sum()

        def gq(x):
            l = x.reshape(I, T)
            return (N - C / l).ravel()

        res = minimize(q, lam.ravel(), jac=gq, method="L-BFGS-B",
                       bounds=[(EPS, None)] * (I * T),
                       options=dict(maxiter=150, ftol=1e-12))
        lam = res.x.reshape(I, T)
        fn = f(lam)
        rel = (fv - fn) / max(abs(fv), 1.0)
        fv = fn
        hist.append((it, time.perf_counter() - t0, fv))
        stall = stall + 1 if rel < 1e-12 else 0
        if stall >= 8:
            break
    return lam, hist


if __name__ == "__main__":
    out = {}
    for name in ["nyc", "chicago", "seattle"]:
        M1, M0, N = build_counts(name)
        f, gr = make_f(M1, M0, N)
        lam0 = np.full((GRID * GRID, T), max(M1.sum() / (N * GRID * GRID * T), 0.05))
        runs = {}
        best = np.inf
        for solver, fn_run in [("PG-Armijo", lambda: run_pg(f, gr, lam0)),
                               ("FISTA", lambda: run_fista(f, gr, lam0)),
                               ("L-BFGS-B", lambda: run_lbfgs(f, gr, lam0, lam0.shape)),
                               ("EM", lambda: run_em(f, M1, M0, N, lam0))]:
            times, hists = [], None
            for rep in range(REPS):
                _, hist = fn_run()
                times.append(hist[-1][1])
                hists = hist
            fend = hists[-1][2]
            best = min(best, fend)
            runs[solver] = dict(iters=int(hists[-1][0]),
                                tempo_mediano_s=round(float(np.median(times)), 2),
                                tempos=[round(t, 2) for t in times],
                                f_final=round(fend, 2), _hist=hists)
        # iteracoes/tempo ate gap relativo 1e-8 vs best
        alvo = best + 1e-8 * abs(best)
        for solver, r in runs.items():
            hit = next(((k, tm) for k, tm, fx in r.pop("_hist") if fx <= alvo),
                       (None, None))
            r["it_ate_1e-8"] = hit[0]
            r["t_ate_1e-8_s"] = round(hit[1], 2) if hit[1] is not None else None
            r["gap_vs_best"] = round(r["f_final"] - best, 4)
        out[name] = dict(N_semanas=round(N, 1), solvers=runs)
        print(name, json.dumps({k: {kk: vv for kk, vv in v.items()}
                                for k, v in runs.items()}), flush=True)
    json.dump(out, open("resultados_bench_solvers.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
