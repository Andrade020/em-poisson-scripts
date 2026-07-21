# -*- coding: utf-8 -*-
"""
Correcao da circularidade apontada pelo referee (M7): o ajuste MNAR agora e
avaliado FORA DA AMOSTRA. Split temporal: p_i e a alocacao-modelo sao
estimados na primeira metade do periodo; a TV do ajuste e avaliada na
distribuicao real dos eventos sem coordenada da SEGUNDA metade.
"""
import json
import numpy as np
from pipeline import load_chicago, load_nyc, prepare

T = 168

def test_oos(df, name, min_zone=100):
    df = df.sort_values("ts")
    cut = df["ts"].quantile(0.5)
    A, B = df[df["ts"] <= cut], df[df["ts"] > cut]

    def counts(dd, zones, zidx):
        rep = dd[(~dd["miss"]) & dd["zone_known"].notna()]
        mis = dd[dd["miss"]]
        mis_lab = mis[mis["zone_known"].notna()]
        M1 = np.zeros((len(zones), T))
        np.add.at(M1, (rep["zone_known"].map(zidx).values, rep["tow"].values), 1)
        M0_t = np.zeros(T)
        np.add.at(M0_t, mis["tow"].values, 1)
        act = np.zeros(len(zones))
        np.add.at(act, mis_lab["zone_known"].map(zidx).values, 1)
        rep_i = np.zeros(len(zones))
        np.add.at(rep_i, rep["zone_known"].map(zidx).values, 1)
        return M1, M0_t, act, rep_i

    zones = sorted(set(df.loc[df["zone_known"].notna(), "zone_known"].unique()))
    zidx = {z: k for k, z in enumerate(zones)}
    M1A, M0A, actA, repA = counts(A, zones, zidx)
    M1B, M0B, actB, repB = counts(B, zones, zidx)

    # p_i estimado SOMENTE na metade A
    p_i = actA / np.maximum(actA + repA, 1)

    # alocacao-modelo na metade B usando o perfil lambda da metade B
    colsum = np.maximum(M1B.sum(0), 1)
    E_model = (M1B / colsum[None, :] * M0B[None, :]).sum(1)
    E_unif = np.full(len(zones), M0B.sum() / len(zones))
    q_act = actB / actB.sum()

    def tv(E):
        return round(0.5 * float(np.abs(E / E.sum() - q_act).sum()), 4)

    odds = np.where((p_i > 0) & (p_i < 1), p_i / (1 - p_i), np.nan)
    odds = np.where(np.isnan(odds), np.nanmedian(odds), odds)
    E_adj = E_model * odds
    return dict(dataset=name,
                faltantes_B=int(actB.sum()),
                TV_modelo=tv(E_model), TV_uniforme=tv(E_unif),
                TV_ajustado_MNAR_oos=tv(E_adj))

if __name__ == "__main__":
    out = []
    out.append(test_oos(prepare(load_chicago()), "chicago_CA"))
    dfn = prepare(load_nyc())
    out.append(test_oos(dfn[(dfn.ts >= "2013-01-01") & (dfn.ts <= "2019-12-31")],
                        "nyc_boroughs_2013-19"))
    for r in out:
        print(json.dumps(r, ensure_ascii=False))
    json.dump(out, open("resultados_validacao_natural_oos.json", "w",
              encoding="utf-8"), indent=1, ensure_ascii=False)
