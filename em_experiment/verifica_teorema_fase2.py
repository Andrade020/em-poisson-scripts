# -*- coding: utf-8 -*-
"""
FASE 2 da falsificacao: os desvios observados nos regimes (i) e (ii) sao o
efeito de amostra finita previsto pela PROPRIA teoria, ou refutacao?

Teste exato: para gamma_N = a_N/N, o alvo de amostra finita e
  th*_gamma = argmin { -Lbar(lam) + (gamma/2) lam'Q lam },
e a previsao refinada e
  E[sqrt(N)(th_hat - th0)] ~ sqrt(N)(th*_gamma - th0),
  Cov[sqrt(N)(th_hat - th*_gamma)] ~ H_g^{-1} Sigma_g H_g^{-1}   (avaliadas em th*_gamma).
Se os z-scores contra ESTA previsao ficarem O(1), os regimes (i)/(ii) estao
confirmados no mecanismo; a formula-limite so difere por gamma_N -> 0.
"""
import json
import numpy as np
from scipy.optimize import minimize
from verifica_teorema import (I, T, D, p0, lam0_nf, Q, d_lam, fisher_lam,
                              simulate, cov_relerr)

def alvo_gamma(gamma):
    S0 = lam0_nf.sum(0)

    def f(x):
        lam = x.reshape(I, T)
        S = lam.sum(0)
        val = (D * S - p0 * S0 * D * np.log(S)).sum() \
            - ((1 - p0)[None, :] * lam0_nf * D[None, :] * np.log(lam)).sum()
        return val + 0.5 * gamma * x @ Q @ x

    def g(x):
        lam = x.reshape(I, T)
        S = lam.sum(0)
        gr = (D - p0 * S0 * D / S)[None, :] - (1 - p0)[None, :] * lam0_nf * D[None, :] / lam
        return gr.ravel() + gamma * (Q @ x)

    res = minimize(f, lam0_nf.ravel(), jac=g, method="L-BFGS-B",
                   bounds=[(1e-9, None)] * d_lam,
                   options=dict(maxiter=5000, ftol=1e-18, gtol=1e-15))
    return res.x.reshape(I, T)

def sanduiche_gamma(lam_star, gamma):
    S_star = lam_star.sum(0)
    S0 = lam0_nf.sum(0)
    Sig = np.zeros((d_lam, d_lam)); H = np.zeros((d_lam, d_lam))
    for t in range(T):
        ix = np.arange(I) * T + t
        varZ = p0[t] * S0[t] * D[t]
        varY = (1 - p0[t]) * lam0_nf[:, t] * D[t]
        Sig[np.ix_(ix, ix)] = varZ / S_star[t] ** 2 * np.ones((I, I)) \
            + np.diag(varY / lam_star[:, t] ** 2)
        H[np.ix_(ix, ix)] = p0[t] * S0[t] * D[t] / S_star[t] ** 2 * np.ones((I, I)) \
            + np.diag((1 - p0[t]) * lam0_nf[:, t] * D[t] / lam_star[:, t] ** 2)
    H = H + gamma * Q
    Hinv = np.linalg.inv(H)
    return Hinv @ Sig @ Hinv

out = {}
cenarios = [("i", None, 3200, lambda n: n ** 0.25),
            ("ii", 1.0, 200, None), ("ii", 1.0, 800, None), ("ii", 1.0, 3200, None),
            ("ii", 4.0, 200, None), ("ii", 4.0, 800, None), ("ii", 4.0, 3200, None)]
for nome, c, N, afn in cenarios:
    if nome == "i":
        aN_fn = afn
        gamma = (N ** 0.25) / N
        seed = 101 + N
        tag = f"i_N={N}"
    else:
        aN_fn = (lambda n, cc=c: cc * np.sqrt(n))
        gamma = c / np.sqrt(N)
        seed = int(211 + N + 10 * c)
        tag = f"ii_c={c}_N={N}"
    lam_star = alvo_gamma(gamma)
    pred = np.sqrt(N) * (lam_star - lam0_nf).ravel()
    lam_h, _ = simulate(lam0_nf, N, 4000, aN_fn, seed=seed)
    U = np.sqrt(N) * (lam_h - lam0_nf[None]).reshape(4000, d_lam)
    emp = U.mean(0)
    se = U.std(0) / np.sqrt(4000)
    z = (emp - pred) / np.maximum(se, 1e-12)
    Ustar = np.sqrt(N) * (lam_h - lam_star[None]).reshape(4000, d_lam)
    cov_pred = sanduiche_gamma(lam_star, gamma)
    out[tag] = dict(
        gamma=round(gamma, 5),
        norma_vies_previsto_finito=round(float(np.linalg.norm(pred)), 3),
        norma_vies_empirico=round(float(np.linalg.norm(emp)), 3),
        corr=round(float(np.corrcoef(emp, pred)[0, 1]), 5),
        max_abs_z_vs_previsao_finita=round(float(np.abs(z).max()), 2),
        cov_relerr_vs_sanduiche_gamma=round(cov_relerr(np.cov(Ustar.T), cov_pred), 4))
    print(tag, json.dumps(out[tag]), flush=True)

json.dump(out, open("resultados_verifica_teorema_fase2.json", "w"), indent=1)
print("FASE 2 CONCLUIDA")
