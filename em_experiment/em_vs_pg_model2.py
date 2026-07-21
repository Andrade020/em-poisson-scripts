# -*- coding: utf-8 -*-
"""
EM penalizado vs. gradiente projetado (Armijo) para o estimador regularizado
do modelo 2 de Guigues, Kleywegt, Nascimento & Andrade (Sci Rep 2026),
"Estimation of spatiotemporal Poisson processes with missing data".

Modelo (um tipo c fixo, omitido):
  Y_{i,t,n} ~ Poisson((1-p_t) lam_{i,t} D_t)   (localizacao reportada)
  Z_{t,n}   ~ Poisson(p_t S_t D_t),  S_t = sum_i lam_{i,t}   (nao reportada)

Estatisticas suficientes: M1[i,t] = sum_n Y, M0[t] = sum_n Z.

Objetivo penalizado (parte em lambda, eq. (15) do artigo):
  l1(lam) = sum_t [ N D_t S_t - M0_t log S_t - sum_i M1_it log lam_it ]
          + sum_i sum_G (W_G/2) sum_{t,t' in G} N^2 (lam_it - lam_it')^2
          + sum_t sum_{i~j} (w/2) N^2 * 2 * (lam_it - lam_jt)^2   [pares nao ordenados]

Metodos comparados:
  1. PG-Armijo: gradiente projetado com busca de Armijo (metodo do artigo/LASPATED)
  2. EM-exato:  E-step aloca M0_t nas zonas com pesos lam/S; M-step resolve o
     problema penalizado de dados completos com L-BFGS-B (convexo, suave)
  3. GEM-5:     EM generalizado, M-step = 5 passos de gradiente projetado
  4. L-BFGS-B direto em l1 (referencia de otimo global; l1 e convexa)

Verificacao adicional: sem penalizacao, o ponto fixo do EM deve coincidir com a
formula fechada (6) do artigo.
"""
import time
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20260717)

# ----------------------------- setup sintetico ------------------------------
GRID = 10                 # grade 10x10 -> I = 100 zonas
I = GRID * GRID
T = 48                    # 48 intervalos de 30 min (1 dia); periodicidade semanal omitida
N = 104                   # numero de observacoes (semanas ~ 2 anos)
D = 0.5 * np.ones(T)      # duracao de cada intervalo (horas)
EPS = 1e-6

# intensidades verdadeiras: zonas de alta demanda (5x5 inferior-esq e superior-dir),
# com ciclo diario senoidal — proximo do experimento de timestamps do artigo
zone_high = np.zeros((GRID, GRID), dtype=bool)
zone_high[:5, :5] = True
zone_high[5:, 5:] = True
zone_high = zone_high.reshape(-1)
tgrid = np.arange(T)
diurnal = 1.0 + 0.6 * np.sin(2 * np.pi * (tgrid - 14) / T)      # pico a tarde
lam_true = np.where(zone_high[:, None], 2.5, 1.0) * diurnal[None, :]   # (I,T)
p_true = 0.25 + 0.2 * np.cos(2 * np.pi * tgrid / T)                    # em [0.05,0.45]

S_true = lam_true.sum(axis=0)
M1 = rng.poisson(N * (1 - p_true)[None, :] * lam_true * D[None, :])    # (I,T)
M0 = rng.poisson(N * p_true * S_true * D)                              # (T,)

# ------------------------------- penalizacao --------------------------------
# grupos de tempo: 4 blocos de 12 intervalos consecutivos
G_groups = [np.arange(g * 12, (g + 1) * 12) for g in range(4)]
# vizinhanca espacial: 4-vizinhos na grade
edges = []
for r in range(GRID):
    for cc in range(GRID):
        k = r * GRID + cc
        if r + 1 < GRID:
            edges.append((k, k + GRID))
        if cc + 1 < GRID:
            edges.append((k, k + 1))
edges = np.array(edges)  # (E,2)

def make_penalty(w):
    """Retorna (pen(lam), grad_pen(lam)) para peso comum w (tempo e espaco).
    Pesos multiplicam N^2 como no artigo (N_{c,t} = N para todo t aqui)."""
    WN2 = w * N * N

    def pen(lam):
        v = 0.0
        for G in G_groups:
            sub = lam[:, G]                       # (I,|G|)
            # sum_{t,t' in G} (x_t - x_t')^2 = 2|G| sum x^2 - 2 (sum x)^2
            v += WN2 * (len(G) * (sub ** 2).sum() - (sub.sum(axis=1) ** 2).sum())
        dif = lam[edges[:, 0], :] - lam[edges[:, 1], :]
        v += WN2 * (dif ** 2).sum()
        return v

    def grad(lam):
        g = np.zeros_like(lam)
        for G in G_groups:
            sub = lam[:, G]
            g[:, G] += 2 * WN2 * (len(G) * sub - sub.sum(axis=1, keepdims=True))
        dif = lam[edges[:, 0], :] - lam[edges[:, 1], :]
        np.add.at(g, edges[:, 0], 2 * WN2 * dif)
        np.add.at(g, edges[:, 1], -2 * WN2 * dif)
        return g

    return pen, grad

def make_objective(w):
    pen, gpen = make_penalty(w)

    def f(lam):
        S = lam.sum(axis=0)
        val = (N * D * S - M0 * np.log(S)).sum() - (M1 * np.log(lam)).sum()
        return val + pen(lam)

    def gradf(lam):
        S = lam.sum(axis=0)
        g = (N * D - M0 / S)[None, :] - M1 / lam
        return g + gpen(lam)

    return f, gradf, pen, gpen

# ------------------------------- otimizadores -------------------------------
def proj(lam):
    return np.maximum(lam, EPS)

def pg_armijo(f, gradf, lam0, max_iter=2000, tol_rel=1e-9, stall=5,
              sigma=1e-4, beta=0.5, t0=None):
    """Gradiente projetado com Armijo ao longo da direcao viavel (como LASPATED).
    Retorna (lam, historia [(iter, tempo, f)], n_feval)."""
    lam = lam0.copy()
    fv = f(lam)
    hist = [(0, 0.0, fv)]
    nfe = 1
    t_start = time.perf_counter()
    step = t0 if t0 is not None else 1.0 / (N * D.max())
    n_stall = 0
    for it in range(1, max_iter + 1):
        g = gradf(lam)
        # busca de Armijo sobre a direcao viavel d = proj(lam - step*g) - lam
        t = step
        for _ in range(60):
            lam_new = proj(lam - t * g)
            d = lam_new - lam
            fn = f(lam_new)
            nfe += 1
            if fn <= fv + sigma * (g * d).sum():
                break
            t *= beta
        rel = (fv - fn) / max(abs(fv), 1.0)
        lam, f_prev, fv = lam_new, fv, fn
        hist.append((it, time.perf_counter() - t_start, fv))
        step = min(t / beta, 1e2 * step)  # expansao leve do passo inicial
        n_stall = n_stall + 1 if rel < tol_rel else 0
        if n_stall >= stall:
            break
    return lam, hist, nfe

def em_penalized(f, lam0, w, max_iter=500, tol_rel=1e-9, stall=5,
                 inner='lbfgs', inner_steps=5):
    """EM penalizado.
    E-step: W_hat[i,t] = M0_t * lam_it / S_t
    M-step: min_lam sum_t [N D_t S_t - sum_i (M1+W_hat) log lam] + pen(lam)
    inner='lbfgs' resolve o M-step com L-BFGS-B; inner='pg' da inner_steps
    passos de gradiente projetado (GEM)."""
    pen, gpen = make_penalty(w)
    lam = lam0.copy()
    fv = f(lam)
    hist = [(0, 0.0, fv)]
    t_start = time.perf_counter()
    n_stall = 0
    for it in range(1, max_iter + 1):
        S = lam.sum(axis=0)
        C = M1 + M0[None, :] * lam / S[None, :]      # dados completos esperados

        def q(x):
            l = x.reshape(I, T)
            return (N * D * l.sum(axis=0)).sum() - (C * np.log(l)).sum() + pen(l)

        def gq(x):
            l = x.reshape(I, T)
            return ((N * D)[None, :] - C / l + gpen(l)).ravel()

        if inner == 'lbfgs':
            res = minimize(q, lam.ravel(), jac=gq, method='L-BFGS-B',
                           bounds=[(EPS, None)] * (I * T),
                           options=dict(maxiter=200, ftol=1e-12, gtol=1e-8))
            lam = res.x.reshape(I, T)
        else:  # GEM: poucos passos de PG no Q
            step = 1.0 / (N * D.max())
            for _ in range(inner_steps):
                g = gq(lam.ravel()).reshape(I, T)
                t = step
                qv = q(lam.ravel())
                for _ in range(40):
                    lam_new = proj(lam - t * g)
                    if q(lam_new.ravel()) <= qv + 1e-4 * (g * (lam_new - lam)).sum():
                        break
                    t *= 0.5
                lam = lam_new
                step = t / 0.5
        f_prev, fv = fv, f(lam)
        hist.append((it, time.perf_counter() - t_start, fv))
        rel = (f_prev - fv) / max(abs(f_prev), 1.0)
        n_stall = n_stall + 1 if rel < tol_rel else 0
        if n_stall >= stall:
            break
    return lam, hist

def lbfgs_direct(f, gradf, lam0):
    t_start = time.perf_counter()
    res = minimize(lambda x: f(x.reshape(I, T)), lam0.ravel(),
                   jac=lambda x: gradf(x.reshape(I, T)).ravel(),
                   method='L-BFGS-B', bounds=[(EPS, None)] * (I * T),
                   options=dict(maxiter=5000, ftol=1e-14, gtol=1e-10))
    return res.x.reshape(I, T), time.perf_counter() - t_start, res

# --------------------------- verificacao sem penalty ------------------------
print("=" * 78)
print("SANIDADE (w = 0): EM sem penalizacao deve reproduzir a formula fechada (6)")
f0, g0, _, _ = make_objective(0.0)
S_hat = (M1.sum(axis=0) + M0) / (N * D)                    # eq. (5)
lam_closed = S_hat[None, :] * M1 / M1.sum(axis=0)[None, :]  # eq. (6)
lam_em0, hist_em0 = em_penalized(f0, np.full((I, T), 1.0), w=0.0, max_iter=3000,
                                 tol_rel=1e-14, stall=10, inner='lbfgs')
err = np.max(np.abs(lam_em0 - lam_closed) / np.maximum(lam_closed, 1e-12))
print(f"  max erro relativo EM vs formula fechada: {err:.2e}  "
      f"({len(hist_em0)-1} iteracoes EM)")
print(f"  l1(EM) - l1(fechada) = {f0(lam_em0) - f0(np.maximum(lam_closed, EPS)):.3e}")

# ------------------------------- comparacao ---------------------------------
def rmse(lam):
    return float(np.sqrt(((lam - lam_true) ** 2).mean()))

results = []
for w in [0.001, 0.01]:
    f, gradf, pen, gpen = make_objective(w)
    lam0 = np.full((I, T), max(M1.sum() / (N * D.sum() * I), 0.5))

    lam_ref, t_ref, res_ref = lbfgs_direct(f, gradf, lam0)
    f_star = f(lam_ref)

    lam_pg, hist_pg, nfe_pg = pg_armijo(f, gradf, lam0, max_iter=3000,
                                        tol_rel=1e-12, stall=8)
    lam_em, hist_em = em_penalized(f, lam0, w, max_iter=400,
                                   tol_rel=1e-12, stall=8, inner='lbfgs')
    lam_gem, hist_gem = em_penalized(f, lam0, w, max_iter=3000,
                                     tol_rel=1e-12, stall=8,
                                     inner='pg', inner_steps=5)

    def report(name, lam, hist, extra=""):
        it, tt, fv = hist[-1]
        gap = fv - f_star
        # iteracoes/tempo para atingir gap relativo 1e-6
        target = f_star + 1e-6 * abs(f_star)
        hit = next(((k, tm) for k, tm, fx in hist if fx <= target), (None, None))
        row = dict(w=w, metodo=name, iters=it, tempo_s=tt, f_final=fv,
                   gap_vs_otimo=gap, it_para_1e6=hit[0], t_para_1e6=hit[1],
                   rmse=rmse(lam))
        results.append(row)

    report("PG-Armijo (artigo)", lam_pg, hist_pg)
    report("EM exato (M-step LBFGS)", lam_em, hist_em)
    report("GEM (M-step 5 passos PG)", lam_gem, hist_gem)
    np.savez(f"hist_w{w}.npz", pg=np.array(hist_pg), em=np.array(hist_em),
             gem=np.array(hist_gem), fstar=f_star)
    results.append(dict(w=w, metodo="L-BFGS-B direto (ref)", iters=res_ref.nit,
                        tempo_s=t_ref, f_final=f_star, gap_vs_otimo=0.0,
                        it_para_1e6=None, t_para_1e6=None, rmse=rmse(lam_ref)))

print()
print("=" * 78)
print(f"COMPARACAO (I={I}, T={T}, N={N}; alvo: gap relativo 1e-6 vs otimo)")
hdr = f"{'w':>6} {'metodo':<26} {'iters':>6} {'tempo(s)':>9} {'gap final':>12} " \
      f"{'it@1e-6':>8} {'t@1e-6':>8} {'RMSE':>7}"
print(hdr)
print("-" * len(hdr))
for r in results:
    it6 = "-" if r['it_para_1e6'] is None else str(r['it_para_1e6'])
    t6 = "-" if r['t_para_1e6'] is None else f"{r['t_para_1e6']:.2f}"
    print(f"{r['w']:>6} {r['metodo']:<26} {r['iters']:>6} {r['tempo_s']:>9.2f} "
          f"{r['gap_vs_otimo']:>12.4e} {it6:>8} {t6:>8} {r['rmse']:>7.4f}")

# RMSE do estimador nao corrigido (descarta M0) para contexto
lam_unc = M1 / ((1 - 0) * N * D)[None, :]  # sem correcao: M1/(N D)
print(f"\nRMSE estimador nao-corrigido (descarta chamadas sem local): "
      f"{rmse(lam_unc):.4f}")
print(f"RMSE estimador fechado corrigido (eq. 6, w=0):               "
      f"{rmse(lam_closed):.4f}")
