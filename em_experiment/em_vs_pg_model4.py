# -*- coding: utf-8 -*-
"""
MCEM (na verdade: EM exato sobre a aproximacao Monte Carlo) vs. gradiente
projetado para o modelo 4 do artigo (probabilidades de localizacao dadas por
covariaveis/populacao, verossimilhanca L4 e objetivo amostrado \hat{l}_4).

Observacao chave: fixadas as amostras multinomiais M^s (common random numbers),
o objetivo do artigo
  \hat{l}_4(lam) = sum_{t} N_t S_t D_t - sum_{t,n} log u_{t,n}(lam),
  u_{t,n}(lam) = e^{-S_t D_t} (1/S) sum_s prod_i (lam_i D)^{a^s_i}/a^s_i!,
  a^s_i = M^s_i + M1_{i,t,n},
e' EXATAMENTE o negativo de uma log-verossimilhanca de mistura finita com S
componentes por observacao. Logo o EM se aplica de forma exata:
  E-step: responsabilidades gamma^s propto exp( sum_i a^s_i log(lam_i D)
                                               - log a^s_i! )
  M-step (fechado): lam_{i,t} = sum_n sum_s gamma^{n,s} a^{n,s}_i / (N_t D_t)
e decresce \hat{l}_4 monotonicamente. Comparamos com gradiente projetado +
Armijo sobre \hat{l}_4 (metodo do artigo, eq. (21)).
"""
import sys
import time
import numpy as np
from scipy.special import gammaln, logsumexp

rng = np.random.default_rng(42)

# ------------------------------ setup ---------------------------------------
# uso: python em_vs_pg_model4.py [I] [T] [N]
I = int(sys.argv[1]) if len(sys.argv) > 1 else 25
T = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
D = 0.5
S_SAMP = 30              # amostras multinomiais por (t,n), como no artigo
EPS = 1e-6

# populacao (define pi) correlacionada mas nao identica a lambda
pop = rng.uniform(0.5, 2.0, size=I)
pi = pop / pop.sum()

zone_high = np.zeros(I, dtype=bool)
zone_high[: I // 2] = True
tfac = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(T) / T)
lam_true = np.where(zone_high[:, None], 2.0, 0.8) * tfac[None, :]   # (I,T)
p_true = 0.30

# geracao (mecanismo de thinning do modelo 2): por observacao n e tempo t,
# X_i ~ Poisson(lam_i D); cada chegada perde o local com prob p
M1 = np.zeros((I, T, N), dtype=int)
M0 = np.zeros((T, N), dtype=int)
for t in range(T):
    for n in range(N):
        X = rng.poisson(lam_true[:, t] * D)
        miss = rng.binomial(X, p_true)
        M1[:, t, n] = X - miss
        M0[t, n] = miss.sum()

# amostras multinomiais fixas (common random numbers), como o LASPATED
# a[s, i, t, n] = M^s_i + M1_{i,t,n}
Msamp = np.zeros((S_SAMP, I, T, N), dtype=int)
for t in range(T):
    for n in range(N):
        if M0[t, n] > 0:
            Msamp[:, :, t, n] = rng.multinomial(M0[t, n], pi, size=S_SAMP)
A = Msamp + M1[None, :, :, :]                      # (S,I,T,N)
LGA = gammaln(A + 1.0)                             # log a!

# ------------------------- objetivo e gradiente -----------------------------
def obj_and_resp(lam):
    """Retorna (\hat{l}_4(lam), responsabilidades gamma (S,T,N),
    contagens esperadas C[i,t] = sum_n sum_s gamma * a)."""
    logl = np.log(lam * D)                                    # (I,T)
    # logw[s,t,n] = sum_i a*log(lam_i D) - log a!
    logw = np.einsum('sitn,it->stn', A, logl) - LGA.sum(axis=1)
    lse = logsumexp(logw, axis=0)                             # (T,N)
    S_t = lam.sum(axis=0)                                     # (T,)
    val = N * D * S_t.sum() - (lse - np.log(S_SAMP)).sum()
    gamma = np.exp(logw - lse[None, :, :])                    # (S,T,N)
    C = np.einsum('stn,sitn->it', gamma, A)                   # (I,T)
    return val, gamma, C

def grad(lam, C):
    # d/dlam_i [N D S_t] = N D ; d/dlam_i [-log u] = -(C_terms/lam - D) por n
    # somando: N D - C/lam + (num de termos com D) ... derivacao:
    # -sum_n dlog u/dlam_i = -sum_n [ E_gamma(a)/lam_i - D ]  (o -D vem do
    # e^{-S D} dentro de u). Com o termo N D S_t fora: N D - N D + ... cuidado:
    # aqui u inclui e^{-S D}? Nao: escrevemos val = N D S - sum log(mixture sem
    # o fator e^{-S D})? Conferir: logw nao inclui -S D, e val soma N D S_t.
    # d val/dlam_it = N D - sum_n E_gamma[a_i]/lam_it = N D - C_it/lam_it.
    return N * D - C / lam

def pg_armijo(lam0, max_iter=5000, tol_rel=1e-11, stall=8):
    lam = lam0.copy()
    fv, _, C = obj_and_resp(lam)
    hist = [(0, 0.0, fv)]
    t0 = time.perf_counter()
    step = 1.0 / (N * D)
    n_stall = 0
    nfe = 1
    for it in range(1, max_iter + 1):
        g = grad(lam, C)
        t = step
        for _ in range(60):
            lam_new = np.maximum(lam - t * g, EPS)
            fn, _, Cn = obj_and_resp(lam_new)
            nfe += 1
            if fn <= fv + 1e-4 * (g * (lam_new - lam)).sum():
                break
            t *= 0.5
        rel = (fv - fn) / max(abs(fv), 1.0)
        lam, fv, C = lam_new, fn, Cn
        hist.append((it, time.perf_counter() - t0, fv))
        step = min(t / 0.5, 100 * step)
        n_stall = n_stall + 1 if rel < tol_rel else 0
        if n_stall >= stall:
            break
    return lam, hist, nfe

def em(lam0, max_iter=5000, tol_rel=1e-11, stall=8):
    lam = lam0.copy()
    fv, _, C = obj_and_resp(lam)
    hist = [(0, 0.0, fv)]
    t0 = time.perf_counter()
    n_stall = 0
    for it in range(1, max_iter + 1):
        lam = np.maximum(C / (N * D), EPS)      # M-step fechado
        fn, _, C = obj_and_resp(lam)
        rel = (fv - fn) / max(abs(fv), 1.0)
        fv = fn
        hist.append((it, time.perf_counter() - t0, fv))
        n_stall = n_stall + 1 if rel < tol_rel else 0
        if n_stall >= stall:
            break
    return lam, hist

# ------------------------------ execucao ------------------------------------
lam0 = np.full((I, T), M1.sum() / (N * D * I * T) + 0.5)

lam_pg, hist_pg, nfe_pg = pg_armijo(lam0)
lam_em, hist_em = em(lam0)

f_pg = hist_pg[-1][2]
f_em = hist_em[-1][2]
f_best = min(f_pg, f_em)

def rmse(lam):
    return float(np.sqrt(((lam - lam_true) ** 2).mean()))

def hit(hist, target):
    return next(((k, tm) for k, tm, fx in hist if fx <= target), (None, None))

target = f_best + 1e-6 * abs(f_best)
hit_pg = hit(hist_pg, target)
hit_em = hit(hist_em, target)

print("=" * 78)
print(f"MODELO 4 (pi por populacao) — I={I}, T={T}, N={N}, S={S_SAMP} amostras")
print(f"  verificacao EM monotono: "
      f"{all(hist_em[k][2] <= hist_em[k-1][2] + 1e-9 for k in range(1, len(hist_em)))}")
print()
hdr = f"{'metodo':<28} {'iters':>6} {'tempo(s)':>9} {'f final':>14} " \
      f"{'gap vs melhor':>13} {'it@1e-6':>8} {'t@1e-6':>8} {'RMSE':>7}"
print(hdr)
print("-" * len(hdr))
for name, lam, histx, extra in [
    ("PG-Armijo (artigo, eq.21)", lam_pg, hist_pg, hit_pg),
    ("EM exato no MC-objetivo", lam_em, hist_em, hit_em),
]:
    it, tt, fv = histx[-1]
    i6 = "-" if extra[0] is None else str(extra[0])
    t6 = "-" if extra[1] is None else f"{extra[1]:.2f}"
    print(f"{name:<28} {it:>6} {tt:>9.2f} {fv:>14.4f} {fv - f_best:>13.4e} "
          f"{i6:>8} {t6:>8} {rmse(lam):>7.4f}")

print(f"\nmax |lam_EM - lam_PG| / lam_PG: "
      f"{np.max(np.abs(lam_em - lam_pg) / lam_pg):.3e}")
lam_disc = M1.sum(axis=2) / (N * D)   # descarta chamadas sem local
print(f"RMSE descartando dados sem local: {rmse(lam_disc):.4f}")
lam_c6 = lam_disc * (1 + M0.sum() / M1.sum())  # correcao global tipo eq. (6)
print(f"RMSE correcao fechada eq.(6):     {rmse(lam_c6):.4f}")
