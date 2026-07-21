# -*- coding: utf-8 -*-
"""
FALSIFICACAO EMPIRICA do Teorema 1 (4 regimes) do paper.

Modelo 2, um tipo, I=6 zonas, T=4 janelas (d_lambda=24, d_p=4). Penalidade
apenas no bloco lambda: Q = Laplaciano do grafo com arestas espaciais em
cadeia (i,i+1) por janela e grupo temporal completo (t,t') por zona; grafo
conexo => ker Q = span{1} (dimensao 1).

Para cada regime, testa a previsao EXATA da distribuicao-limite contra
Monte Carlo com R replicas, reportando discrepancias padronizadas (z) com
erro de MC. Criterio de refutacao: |z| > 4 em vies previsto, ou erro
relativo de covariancia que NAO decresce com N.

Cenarios:
  (i)    a_N = N^{1/4}          verdade nao fundida: vies*sqrt(N)->0, cov->I^{-1}, cobertura 90%
  (ii)   a_N = c*sqrt(N), c=1,4 vies de sqrt(N)(th-th0) -> -c*I^{-1}Q th0 (componente a componente; linearidade em c)
  (iii-a) a_N = a*N, a=0.5      verdade fundida: cov -> (I+aQ)^{-1} I (I+aQ)^{-1}  (e difere de I^{-1})
  (iii-b) a_N = a*N, a=0.5      verdade NAO fundida: th -> th*_a (calculado); cov de sqrt(N)(th-th*_a) -> Hbar^{-1} Sigma Hbar^{-1}
  (iv-a)  a_N = w*N^2           verdade fundida: componentes ortogonais a ker Q colapsam; cov projetada -> Pi(Pi' I Pi)^{-1} Pi'
  (iv-b)  a_N = w*N^2           verdade nao fundida: ||th-th0|| NAO -> 0; dist(th, ker Q) -> 0
  (int)   a_N = N^{3/4}         verdade nao fundida: erro nas direcoes penalizadas ~ N/a_N = N^{1/4} (consistente, taxa lenta)

Solver: bloco p em forma fechada (sem penalidade em p); bloco lambda por
L-BFGS-B (objetivo convexo), warm start na verdade.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(20260721)
I, T = 6, 4
D = np.array([1.0, 0.5, 1.0, 2.0])
d_lam = I * T

# --------------------- verdade (nao fundida) e fundida -----------------------
lam0_nf = (0.8 + 1.4 * rng.random((I, T)))          # nao fundida
lam0_f = np.full((I, T), 1.5)                        # fundida (ker Q)
p0 = np.array([0.15, 0.35, 0.25, 0.4])

# ------------------------------- Q (Laplaciano) ------------------------------
def build_Q():
    Q = np.zeros((d_lam, d_lam))
    idx = lambda i, t: i * T + t
    for t in range(T):                    # cadeia espacial
        for i in range(I - 1):
            a, b = idx(i, t), idx(i + 1, t)
            Q[a, a] += 1; Q[b, b] += 1; Q[a, b] -= 1; Q[b, a] -= 1
    for i in range(I):                    # grupo temporal completo
        for t in range(T):
            for u in range(t + 1, T):
                a, b = idx(i, t), idx(i, u)
                Q[a, a] += 1; Q[b, b] += 1; Q[a, b] -= 1; Q[b, a] -= 1
    return Q

Q = build_Q()
evals = np.linalg.eigvalsh(Q)
assert evals[0] < 1e-10 and evals[1] > 1e-10        # ker Q = span{1}
Pi = np.ones((d_lam, 1)) / np.sqrt(d_lam)

# --------------------------- Fisher (bloco lambda) ---------------------------
def fisher_lam(lam, p):
    """I_lam (d_lam x d_lam), bloco-diagonal em t (eq. fisher do paper)."""
    F = np.zeros((d_lam, d_lam))
    for t in range(T):
        S = lam[:, t].sum()
        blk = (p[t] * D[t] / S) * np.ones((I, I)) + \
            np.diag((1 - p[t]) * D[t] / lam[:, t])
        ix = np.arange(I) * T + t
        F[np.ix_(ix, ix)] = blk
    return F

# ------------------------------- estimador -----------------------------------
def fit(M1, M0, N, aN, x0):
    """p fechado; lambda por L-BFGS-B em F(lam) = -L + (aN/2) lam'Q lam."""
    M1s = M1.sum(0)
    p_hat = M0 / np.maximum(M0 + M1s, 1e-12)

    def f(x):
        lam = x.reshape(I, T)
        S = lam.sum(0)
        val = (N * D * S - M0 * np.log(S)).sum() - (M1 * np.log(lam)).sum()
        return val + 0.5 * aN * x @ Q @ x

    def g(x):
        lam = x.reshape(I, T)
        S = lam.sum(0)
        gr = (N * D - M0 / S)[None, :] - M1 / lam
        return gr.ravel() + aN * (Q @ x)

    res = minimize(f, x0, jac=g, method="L-BFGS-B",
                   bounds=[(1e-9, None)] * d_lam,
                   options=dict(maxiter=500, ftol=1e-14, gtol=1e-12))
    return res.x.reshape(I, T), p_hat

def simulate(lam0, N, R, aN_fn, seed):
    """R replicas; retorna arrays lam_hat (R,I,T), p_hat (R,T)."""
    rg = np.random.default_rng(seed)
    S0 = lam0.sum(0)
    aN = aN_fn(N)
    x0 = lam0.ravel().copy()
    lam_hats = np.empty((R, I, T)); p_hats = np.empty((R, T))
    for r in range(R):
        M1 = rg.poisson(N * (1 - p0)[None, :] * lam0 * D[None, :]).astype(float)
        M0 = rg.poisson(N * p0 * S0 * D).astype(float)
        lam_hats[r], p_hats[r] = fit(M1, M0, N, aN, x0)
    return lam_hats, p_hats

def zstats(emp_mean, pred_mean, emp_sd, R):
    se = emp_sd / np.sqrt(R)
    return (emp_mean - pred_mean) / np.maximum(se, 1e-12)

def cov_relerr(emp_cov, pred_cov):
    return float(np.linalg.norm(emp_cov - pred_cov) / np.linalg.norm(pred_cov))

resultados = {}
t_ini = time.perf_counter()

# ============================ REGIME (i) =====================================
reg = {}
for N in [50, 200, 800, 3200]:
    R = 4000
    lam_h, p_h = simulate(lam0_nf, N, R, lambda n: n ** 0.25, seed=101 + N)
    U = np.sqrt(N) * (lam_h - lam0_nf[None]).reshape(R, d_lam)
    Ilam = fisher_lam(lam0_nf, p0)
    Iinv = np.linalg.inv(Ilam)
    z = zstats(U.mean(0), np.zeros(d_lam), U.std(0), R)
    # cobertura IC 90% para lambda via [I(th_hat)^{-1}]_jj / N
    cover = 0.0
    for r in range(min(R, 1000)):
        Fh = fisher_lam(lam_h[r], p_h[r])
        se = np.sqrt(np.diag(np.linalg.inv(Fh)) / N)
        lo = lam_h[r].ravel() - 1.6449 * se
        hi = lam_h[r].ravel() + 1.6449 * se
        cover += ((lam0_nf.ravel() >= lo) & (lam0_nf.ravel() <= hi)).mean() / min(R, 1000)
    reg[N] = dict(max_abs_z_vies=round(float(np.abs(z).max()), 2),
                  cov_relerr_vs_Iinv=round(cov_relerr(np.cov(U.T), Iinv), 4),
                  cobertura_IC90=round(float(cover), 4))
resultados["regime_i"] = reg
print("regime i:", json.dumps(reg), flush=True)

# ============================ REGIME (ii) ====================================
reg = {}
Ilam = fisher_lam(lam0_nf, p0)
Iinv = np.linalg.inv(Ilam)
for c in [1.0, 4.0]:
    for N in [200, 800, 3200]:
        R = 4000
        lam_h, _ = simulate(lam0_nf, N, R, lambda n: c * np.sqrt(n), seed=int(211 + N + 10 * c))
        U = np.sqrt(N) * (lam_h - lam0_nf[None]).reshape(R, d_lam)
        shift_pred = -c * Iinv @ (Q @ lam0_nf.ravel())
        z = zstats(U.mean(0), shift_pred, U.std(0), R)
        corr = float(np.corrcoef(U.mean(0), shift_pred)[0, 1])
        reg[f"c={c}_N={N}"] = dict(
            corr_vies_emp_vs_teoria=round(corr, 4),
            max_abs_z=round(float(np.abs(z).max()), 2),
            norma_vies_teoria=round(float(np.linalg.norm(shift_pred)), 3),
            norma_vies_emp=round(float(np.linalg.norm(U.mean(0))), 3),
            cov_relerr_vs_Iinv=round(cov_relerr(np.cov(U.T), Iinv), 4))
        print(f"regime ii c={c} N={N}:", json.dumps(reg[f'c={c}_N={N}']), flush=True)
resultados["regime_ii"] = reg

# =========================== REGIME (iii-a) ==================================
reg = {}
a = 0.5
Ilam_f = fisher_lam(lam0_f, p0)
Mat = np.linalg.inv(Ilam_f + a * Q)
cov_pred = Mat @ Ilam_f @ Mat
for N in [200, 800, 3200]:
    R = 4000
    lam_h, _ = simulate(lam0_f, N, R, lambda n: a * n, seed=331 + N)
    U = np.sqrt(N) * (lam_h - lam0_f[None]).reshape(R, d_lam)
    reg[N] = dict(
        max_abs_z_vies=round(float(np.abs(zstats(U.mean(0), np.zeros(d_lam), U.std(0), R)).max()), 2),
        cov_relerr_vs_sanduiche=round(cov_relerr(np.cov(U.T), cov_pred), 4),
        cov_relerr_vs_Iinv_ingenuo=round(cov_relerr(np.cov(U.T), np.linalg.inv(Ilam_f)), 4))
    print(f"regime iii-a N={N}:", json.dumps(reg[N]), flush=True)
resultados["regime_iii_a"] = reg

# =========================== REGIME (iii-b) ==================================
# alvo penalizado th*_a: min f_a(lam) = sum_t [D S - p0 S0 D log S
#                        - sum_i (1-p0) lam0 D log lam] + (a/2) lam'Q lam
def f_pop(x):
    lam = x.reshape(I, T)
    S = lam.sum(0)
    S0 = lam0_nf.sum(0)
    val = (D * S - p0 * S0 * D * np.log(S)).sum() \
        - ((1 - p0)[None, :] * lam0_nf * D[None, :] * np.log(lam)).sum()
    return val + 0.5 * a * x @ Q @ x

def g_pop(x):
    lam = x.reshape(I, T)
    S = lam.sum(0)
    S0 = lam0_nf.sum(0)
    gr = (D - p0 * S0 * D / S)[None, :] - (1 - p0)[None, :] * lam0_nf * D[None, :] / lam
    return gr.ravel() + a * (Q @ x)

res = minimize(f_pop, lam0_nf.ravel(), jac=g_pop, method="L-BFGS-B",
               bounds=[(1e-9, None)] * d_lam,
               options=dict(maxiter=2000, ftol=1e-16, gtol=1e-14))
lam_star = res.x.reshape(I, T)
dist_star = float(np.linalg.norm(lam_star - lam0_nf))

# Sigma_a (cov do score por periodo em th*_a) e Hbar_a analiticos
S_star = lam_star.sum(0)
S0v = lam0_nf.sum(0)
Sigma = np.zeros((d_lam, d_lam)); Hbar = np.zeros((d_lam, d_lam))
for t in range(T):
    ix = np.arange(I) * T + t
    varZ = p0[t] * S0v[t] * D[t]
    varY = (1 - p0[t]) * lam0_nf[:, t] * D[t]
    Sigma[np.ix_(ix, ix)] = varZ / S_star[t] ** 2 * np.ones((I, I)) \
        + np.diag(varY / lam_star[:, t] ** 2)
    Hbar[np.ix_(ix, ix)] = p0[t] * S0v[t] * D[t] / S_star[t] ** 2 * np.ones((I, I)) \
        + np.diag((1 - p0[t]) * lam0_nf[:, t] * D[t] / lam_star[:, t] ** 2)
Hbar = Hbar + a * Q
Hinv = np.linalg.inv(Hbar)
cov_pred_b = Hinv @ Sigma @ Hinv

reg = {}
for N in [200, 800, 3200]:
    R = 4000
    lam_h, _ = simulate(lam0_nf, N, R, lambda n: a * n, seed=441 + N)
    Ustar = np.sqrt(N) * (lam_h - lam_star[None]).reshape(R, d_lam)
    dist_emp = float(np.linalg.norm(lam_h.mean(0) - lam_star.ravel().reshape(I, T)))
    reg[N] = dict(
        dist_thhat_a_thstar=round(dist_emp, 4),
        dist_thstar_a_th0=round(dist_star, 4),
        max_abs_z_vies_em_thstar=round(float(np.abs(zstats(Ustar.mean(0), np.zeros(d_lam), Ustar.std(0), R)).max()), 2),
        cov_relerr_vs_sanduiche=round(cov_relerr(np.cov(Ustar.T), cov_pred_b), 4))
    print(f"regime iii-b N={N}:", json.dumps(reg[N]), flush=True)
resultados["regime_iii_b"] = reg

# ============================ REGIME (iv-a) ==================================
reg = {}
w = 0.01
Ilam_f = fisher_lam(lam0_f, p0)
cov_ker_pred = Pi @ np.linalg.inv(Pi.T @ Ilam_f @ Pi) @ Pi.T
for N in [200, 800, 3200]:
    R = 4000
    lam_h, _ = simulate(lam0_f, N, R, lambda n: w * n ** 2, seed=551 + N)
    U = np.sqrt(N) * (lam_h - lam0_f[None]).reshape(R, d_lam)
    U_ker = U @ Pi          # componente em ker Q
    U_perp = U - U_ker @ Pi.T
    var_ker_emp = float(U_ker.var())
    var_ker_pred = float((Pi.T @ cov_ker_pred @ Pi))
    reg[N] = dict(
        var_kerQ_emp=round(var_ker_emp, 5),
        var_kerQ_teoria=round(var_ker_pred, 5),
        razao=round(var_ker_emp / var_ker_pred, 3),
        var_ortogonal_media=round(float(U_perp.var(0).mean()), 5))
    print(f"regime iv-a N={N}:", json.dumps(reg[N]), flush=True)
resultados["regime_iv_a"] = reg

# ============================ REGIME (iv-b) ==================================
reg = {}
for N in [200, 800, 3200]:
    R = 1000
    lam_h, _ = simulate(lam0_nf, N, R, lambda n: w * n ** 2, seed=661 + N)
    err0 = np.linalg.norm(lam_h.reshape(R, d_lam) - lam0_nf.ravel()[None], axis=1)
    proj = lam_h.reshape(R, d_lam) @ Pi @ Pi.T
    dker = np.linalg.norm(lam_h.reshape(R, d_lam) - proj, axis=1)
    reg[N] = dict(dist_a_th0_media=round(float(err0.mean()), 4),
                  dist_a_kerQ_media=round(float(dker.mean()), 4))
    print(f"regime iv-b N={N}:", json.dumps(reg[N]), flush=True)
resultados["regime_iv_b"] = reg

# ========================= REGIME intermediario ==============================
reg = {}
for N in [200, 800, 3200, 12800]:
    R = 1000
    lam_h, _ = simulate(lam0_nf, N, R, lambda n: n ** 0.75, seed=771 + N)
    err = np.linalg.norm(lam_h.reshape(R, d_lam) - lam0_nf.ravel()[None], axis=1)
    reg[N] = dict(dist_a_th0_media=round(float(err.mean()), 4),
                  taxa_esperada_Ninv4=round(N ** -0.25, 4))
    print(f"regime int N={N}:", json.dumps(reg[N]), flush=True)
resultados["regime_intermediario"] = reg

resultados["tempo_total_min"] = round((time.perf_counter() - t_ini) / 60, 1)
json.dump(resultados, open("resultados_verifica_teorema.json", "w"), indent=1)
print("CONCLUIDO em", resultados["tempo_total_min"], "min")
