# -*- coding: utf-8 -*-
"""
Analise dos dados REAIS do artigo (Rio de Janeiro EMS, 2016-2018), lidos do
repositorio de replicacao LASPATED (Missing_Data/Rect10x10).

Formatos (conferidos em aggregate_missing.py e nos .dat):
  arrivals.dat:      t g i p n count   (t: 0-47 janelas de 30min, g: 0-6 dia,
                     i: zona 0-99 da grade 10x10, p: prioridade 0-2,
                     n: indice da observacao/semana, count: chegadas reportadas)
  missing_calls.dat: t g -1 p n count [flag]  (chegadas SEM localizacao)

Produz:
  1. p-hat_{c,t} (probabilidade de nao reportar local) por janela da semana e
     prioridade — reproducao da Fig. 3 do artigo (numerica).
  2. Subestimacao por descarte (global e por prioridade).
  3. EM penalizado vs gradiente projetado no problema regularizado real
     (T=336, I=100, C=3), pesos {0, 1e-5, 1e-4}.
  4. Validacao out-of-sample por semanas (ultimas ~20% semanas): prever
     contagens reportadas por (i,t) e totais (reportado+nao) por t.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize

BASE = "LASPATED_Replication/Missing_Data/Rect10x10"
T48, G7, I, P = 48, 7, 100, 3
T = T48 * G7          # 336 janelas da semana
EPS = 1e-6
D = 0.5               # meia hora


def load():
    arr = {}
    nmax = 0
    for line in open(f"{BASE}/arrivals.dat"):
        line = line.strip()
        if line == "END":
            break
        if not line:
            continue
        tok = line.split()
        if len(tok) < 6:
            continue
        t, g, i, p, n, c = (int(tok[0]), int(float(tok[1])), int(tok[2]),
                            int(tok[3]), int(tok[4]), int(tok[5]))
        arr[(t, g, i, p, n)] = c
        nmax = max(nmax, n + 1)
    mis = {}
    for line in open(f"{BASE}/missing_calls.dat"):
        line = line.strip()
        if not line or line == "END":
            continue
        tok = line.split()
        if len(tok) < 6:
            continue
        t, g, p, n, c = (int(tok[0]), int(float(tok[1])), int(tok[3]),
                         int(tok[4]), int(tok[5]))
        mis[(t, g, p, n)] = c
        nmax = max(nmax, n + 1)
    N = nmax
    M1 = np.zeros((P, I, T, N))
    M0 = np.zeros((P, T, N))
    for (t, g, i, p, n), c in arr.items():
        M1[p, i, g * T48 + t, n] = c
    for (t, g, p, n), c in mis.items():
        M0[p, g * T48 + t, n] = c
    return M1, M0, N


def em_vs_pg(M1s, M0s, N, w):
    """Modelo 2 penalizado para um tipo: M1s (I,T), M0s (T,). Retorna dict."""
    edges = []
    for r in range(10):
        for c in range(10):
            k = r * 10 + c
            if r + 1 < 10:
                edges.append((k, k + 10))
            if c + 1 < 10:
                edges.append((k, k + 1))
    edges = np.array(edges)
    # grupos de tempo: mesma meia-hora, dias uteis (seg-sex) e fds (sab-dom)
    groups = [[d_ * T48 + t for d_ in range(5)] for t in range(T48)] + \
             [[d_ * T48 + t for d_ in (5, 6)] for t in range(T48)]
    WN2 = w * N * N

    def pen_grad(lam):
        v, g = 0.0, np.zeros_like(lam)
        for G in groups:
            sub = lam[:, G]
            v += WN2 * (len(G) * (sub ** 2).sum() - (sub.sum(1) ** 2).sum())
            g[:, G] += 2 * WN2 * (len(G) * sub - sub.sum(1, keepdims=True))
        dif = lam[edges[:, 0]] - lam[edges[:, 1]]
        v += WN2 * (dif ** 2).sum()
        np.add.at(g, edges[:, 0], 2 * WN2 * dif)
        np.add.at(g, edges[:, 1], -2 * WN2 * dif)
        return v, g

    def f(lam):
        S = lam.sum(0)
        return float((N * D * S - M0s * np.log(S)).sum()
                     - (M1s * np.log(lam)).sum() + pen_grad(lam)[0])

    def gr(lam):
        S = lam.sum(0)
        return (N * D - M0s / S)[None, :] - M1s / lam + pen_grad(lam)[1]

    lam0 = np.full((I, T), max(M1s.sum() / (N * D * I * T), 0.05))

    # PG-Armijo
    t0 = time.perf_counter()
    lam, fv, step, stall, it_pg = lam0.copy(), None, 1.0 / (N * D), 0, 0
    fv = f(lam)
    for it in range(1, 3001):
        g = gr(lam)
        ts = step
        for _ in range(50):
            ln = np.maximum(lam - ts * g, EPS)
            fn = f(ln)
            if fn <= fv + 1e-4 * (g * (ln - lam)).sum():
                break
            ts *= 0.5
        rel = (fv - fn) / max(abs(fv), 1.0)
        lam, fv, step = ln, fn, ts / 0.5
        stall = stall + 1 if rel < 1e-10 else 0
        it_pg = it
        if stall >= 6:
            break
    t_pg, lam_pg, f_pg = time.perf_counter() - t0, lam, fv

    # EM penalizado
    t0 = time.perf_counter()
    lam, fv, stall, it_em = lam0.copy(), f(lam0), 0, 0
    for it in range(1, 301):
        S = lam.sum(0)
        C = M1s + M0s[None, :] * lam / S[None, :]

        def q(x):
            l = x.reshape(I, T)
            return (N * D * l.sum(0)).sum() - (C * np.log(l)).sum() + pen_grad(l)[0]

        def gq(x):
            l = x.reshape(I, T)
            return ((N * D) - C / l + pen_grad(l)[1]).ravel()

        res = minimize(q, lam.ravel(), jac=gq, method="L-BFGS-B",
                       bounds=[(EPS, None)] * (I * T),
                       options=dict(maxiter=150, ftol=1e-12))
        lam = res.x.reshape(I, T)
        fn = f(lam)
        rel = (fv - fn) / max(abs(fv), 1.0)
        fv, it_em = fn, it
        stall = stall + 1 if rel < 1e-10 else 0
        if stall >= 6:
            break
    t_em, lam_em, f_em = time.perf_counter() - t0, lam, fv
    return dict(PG=dict(iters=it_pg, tempo_s=round(t_pg, 1), f=round(f_pg, 2)),
                EM=dict(iters=it_em, tempo_s=round(t_em, 1), f=round(f_em, 2)),
                gap_EM_menos_PG=round(f_em - f_pg, 4)), lam_em


if __name__ == "__main__":
    M1, M0, N = load()
    R = dict(N_observacoes=N)
    R["total_reportadas"] = int(M1.sum())
    R["total_sem_local"] = int(M0.sum())
    R["p_global"] = round(float(M0.sum() / (M0.sum() + M1.sum())), 4)

    # p-hat por prioridade e estrutura temporal (reproducao da Fig. 3)
    prios = ["alta(0)", "intermediaria(1)", "baixa(2)"]
    R["p_por_prioridade"] = {}
    for p in range(P):
        m0, m1 = M0[p].sum(), M1[p].sum()
        pt = M0[p].sum(1) / np.maximum(M0[p].sum(1) + M1[p].sum((0, 2)), 1)
        R["p_por_prioridade"][prios[p]] = dict(
            p_hat=round(float(m0 / (m0 + m1)), 4),
            p_t_min=round(float(pt.min()), 4), p_t_max=round(float(pt.max()), 4))

    # subestimacao por descarte
    R["subestimacao_%_por_prioridade"] = {
        prios[p]: round(100 * float(M0[p].sum() / max(M1[p].sum(), 1)), 1)
        for p in range(P)}

    # OOS: ultimas 20% semanas como teste
    ntest = max(4, N // 5)
    tr = slice(0, N - ntest)
    te = slice(N - ntest, N)
    Ntr, Nte = N - ntest, ntest
    R["EM_vs_PG"] = {}
    R["OOS"] = {}
    for p in range(P):
        M1tr, M0tr = M1[p, :, :, tr].sum(2), M0[p, :, tr].sum(1)
        M1te, M0te = M1[p, :, :, te].sum(2), M0[p, :, te].sum(1)
        for w in (0.0, 1e-5):
            key = f"prio{p}_w{w}"
            res, lam_hat = em_vs_pg(M1tr, M0tr, Ntr, w)
            R["EM_vs_PG"][key] = res
            # previsao dos totais por t no teste (reportado+nao): S_hat*D*Nte
            S_hat = lam_hat.sum(0)
            tot_te = M1te.sum(0) + M0te
            pred_tot = S_hat * D * Nte
            lam_unc = M1tr / (Ntr * D)          # descarta sem-local
            pred_unc = lam_unc.sum(0) * D * Nte
            mse_c = float(((pred_tot - tot_te) ** 2).mean())
            mse_u = float(((pred_unc - tot_te) ** 2).mean())
            # previsao dos reportados por (i,t): lam*(1-p_t)
            p_t = M0tr / np.maximum(M0tr + M1tr.sum(0), 1)
            pred_rep = lam_hat * (1 - p_t)[None, :] * D * Nte
            mse_rep = float(((pred_rep - M1te) ** 2).mean())
            mse_rep_emp = float(((M1tr / Ntr * Nte - M1te) ** 2).mean())
            R["OOS"][key] = dict(
                MSE_total_corrigido=round(mse_c, 2),
                MSE_total_descartando=round(mse_u, 2),
                razao_unc_sobre_corr=round(mse_u / max(mse_c, 1e-12), 2),
                MSE_reportados_modelo=round(mse_rep, 4),
                MSE_reportados_empirico=round(mse_rep_emp, 4))
    print(json.dumps(R, indent=1, ensure_ascii=False))
    with open("resultados_rio.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)
