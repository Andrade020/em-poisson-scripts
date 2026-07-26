# -*- coding: utf-8 -*-
"""
Teste de permutacao estratificado para dependencia de zona da probabilidade
de faltante (Chicago). Estatistica: deviance binomial do modelo com p_i por
zona contra p constante, calculada sobre os eventos com rotulo de zona
(reportados via coordenada + faltantes com community area do endereco).
Nulo: dado o estrato hora-da-semana, a flag de faltante independe da zona;
permutamos as flags DENTRO de cada estrato tow, preservando p_t.
Saida: resultados_perm_mnar.json (deviance observada, excedencias, p-valor).
"""
import json
import numpy as np
from pipeline import load_chicago, prepare

T = 168
R = 200

def deviance_zone(m_i, n_i):
    # razao de verossimilhanca binomial: H0 p constante vs p_i livre (I-1 gl)
    p0 = m_i.sum() / n_i.sum()
    k_i = n_i - m_i
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(m_i > 0, m_i * np.log(m_i / (n_i * p0)), 0.0)
        t2 = np.where(k_i > 0, k_i * np.log(k_i / (n_i * (1 - p0))), 0.0)
    return 2.0 * float((t1 + t2).sum())

if __name__ == "__main__":
    df = prepare(load_chicago())
    lab = df[df["zone_known"].notna()]
    zones = sorted(lab["zone_known"].unique())
    zidx = {z: k for k, z in enumerate(zones)}
    zi = lab["zone_known"].map(zidx).to_numpy()
    tow = lab["tow"].to_numpy()
    miss = lab["miss"].to_numpy().astype(bool)
    I = len(zones)
    n_i = np.bincount(zi, minlength=I).astype(float)

    def dev(flags):
        m_i = np.bincount(zi[flags], minlength=I).astype(float)
        return deviance_zone(m_i, n_i)

    d_obs = dev(miss)
    print(f"zonas={I}  gl={I-1}  deviance observada={d_obs:.1f}", flush=True)

    strata = [np.where(tow == t)[0] for t in range(T)]
    rng = np.random.default_rng(0)
    exc = 0
    for r in range(R):
        perm = miss.copy()
        for idx in strata:
            perm[idx] = rng.permutation(miss[idx])
        if dev(perm) >= d_obs:
            exc += 1
    out = dict(deviance_observada=round(d_obs, 1), permutacoes=R,
               excedencias=int(exc), p_valor_perm=round((exc + 1) / (R + 1), 3))
    print(json.dumps(out, indent=1))
    json.dump(out, open("resultados_perm_mnar.json", "w"), indent=1)
