# -*- coding: utf-8 -*-
"""
Simulacao de cobertura dos ICs de Fisher para LAMBDA e p (referee M8):
modelo 2 totalmente sintetico com verdade conhecida, R replicas; estimador
fechado (eq. 6) e ICs 90% via inversa da informacao (bloco lambda invertido
em forma fechada por Sherman-Morrison: I_lam = (pD/S) ee' + diag((1-p)D/lam)).
"""
import json
import numpy as np

rng = np.random.default_rng(2026)
I, T, N, D = 25, 24, 104, 1.0
R = 1000
Z90 = 1.6449

lam = 0.5 + 1.5 * rng.random((I, T))          # verdade
p = 0.15 + 0.25 * rng.random(T)
S = lam.sum(0)

cov_lam, cov_p = [], []
for _ in range(R):
    M1 = rng.poisson(N * (1 - p)[None, :] * lam * D)
    M0 = rng.poisson(N * p * S * D)
    M1s = M1.sum(0)
    tot = M1s + M0
    p_hat = M0 / np.maximum(tot, 1)
    S_hat = tot / (N * D)
    lam_hat = S_hat[None, :] * M1 / np.maximum(M1s, 1)[None, :]

    # IC para p: Var = p(1-p)/(N D S) avaliado no estimado
    se_p = np.sqrt(p_hat * (1 - p_hat) / np.maximum(tot, 1))
    cov_p.append(((p >= p_hat - Z90 * se_p) & (p <= p_hat + Z90 * se_p)).mean())

    # IC para lambda: diag de I^{-1}/N com Sherman-Morrison por janela t
    # I_lam(t) = a ee' + diag(d_i), a = p D/S, d_i = (1-p) D/lam_i
    a = p_hat * D / np.maximum(S_hat, 1e-9)
    dinv = lam_hat / np.maximum((1 - p_hat)[None, :] * D, 1e-12)   # 1/d_i
    denom = 1 + a[None, :] * dinv.sum(0)[None, :]
    var_lam = (dinv - a[None, :] * dinv ** 2 / denom) / N
    se_l = np.sqrt(np.maximum(var_lam, 0))
    cov_lam.append(((lam >= lam_hat - Z90 * se_l) &
                    (lam <= lam_hat + Z90 * se_l)).mean())

out = dict(config=dict(I=I, T=T, N=N, R=R, nivel=0.90),
           cobertura_lambda=round(float(np.mean(cov_lam)), 4),
           se_mc_lambda=round(float(np.std(cov_lam) / np.sqrt(R)), 4),
           cobertura_p=round(float(np.mean(cov_p)), 4),
           se_mc_p=round(float(np.std(cov_p) / np.sqrt(R)), 4))
print(json.dumps(out, indent=1))
json.dump(out, open("resultados_coverage_sim.json", "w"), indent=1)
