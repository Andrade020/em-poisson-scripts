# -*- coding: utf-8 -*-
"""
VALIDACAO 100% NATURAL (nada imposto): teste direto da alocacao espacial do
modelo em eventos com localizacao genuinamente faltante.

Design: em Chicago (e NYC), as linhas SEM coordenada tem, na quase totalidade,
o rotulo de zona (community area / borough) — que veio do endereco, nao do
geocode. Entao:
  - o ESTIMADOR ve apenas o que o modelo do artigo veria: eventos reportados
    com zona (via coordenada) + contagens de eventos sem localizacao por
    janela de tempo;
  - a VERDADE (zona real dos eventos "sem localizacao") existe para validacao,
    pelo rotulo que o estimador nao usa.
O mecanismo de faltante e o real (MNAR e tudo). Testamos a implicacao central
do modelo de thinning: eventos faltantes distribuem-se no espaco como
lambda_{i,t} (alocacao proporcional as intensidades estimadas).

Metricas:
  - TV (dist. de variacao total) entre a distribuicao espacial prevista e a
    real dos eventos faltantes; baselines: alocacao estatica (ignora t) e
    uniforme.
  - Assinatura MNAR: razao real/previsto por zona deve ser ~ p_i/(1-p_i)
    (zonas que mais escondem localizacao recebem mais faltantes do que a
    alocacao MAR preve). Reportamos a correlacao dessa previsao teorica.
"""
import json
import numpy as np
import pandas as pd
from pipeline import load_chicago, load_nyc, prepare

T = 168

def test_allocation(df, name, min_zone=200):
    rep = df[(~df["miss"]) & df["zone_known"].notna()]
    mis = df[df["miss"]]
    mis_lab = mis[mis["zone_known"].notna()]
    out = dict(dataset=name,
               n_reportados=int(len(rep)),
               n_faltantes=int(len(mis)),
               frac_faltantes_com_rotulo=round(len(mis_lab) / max(len(mis), 1), 4))

    zones = sorted(set(rep["zone_known"].unique()) | set(mis_lab["zone_known"].unique()))
    zidx = {z: k for k, z in enumerate(zones)}
    I = len(zones)

    M1 = np.zeros((I, T))
    np.add.at(M1, (rep["zone_known"].map(zidx).values, rep["tow"].values), 1)
    M0_t = np.zeros(T)
    np.add.at(M0_t, mis["tow"].values, 1)

    # alocacao do modelo (thinning MAR): E_i = sum_t M0_t * M1[i,t]/M1[.,t]
    colsum = np.maximum(M1.sum(0), 1)
    E_model = (M1 / colsum[None, :] * M0_t[None, :]).sum(1)
    # baselines
    E_static = M1.sum(1) / M1.sum() * M0_t.sum()
    E_unif = np.full(I, M0_t.sum() / I)

    actual = np.zeros(I)
    np.add.at(actual, mis_lab["zone_known"].map(zidx).values, 1)

    q_act = actual / actual.sum()
    def tv(E):
        return round(0.5 * float(np.abs(E / E.sum() - q_act).sum()), 4)
    out["TV_modelo_lambda_t"] = tv(E_model)
    out["TV_alocacao_estatica"] = tv(E_static)
    out["TV_uniforme"] = tv(E_unif)

    # assinatura MNAR: razao observado/previsto vs p_i/(1-p_i)
    tot_i = np.zeros(I)
    np.add.at(tot_i, rep["zone_known"].map(zidx).values, 1)
    mis_i = np.zeros(I)
    np.add.at(mis_i, mis_lab["zone_known"].map(zidx).values, 1)
    p_i = mis_i / np.maximum(mis_i + tot_i, 1)
    ok = (actual >= min_zone) & (E_model > 0) & (p_i > 0) & (p_i < 1)
    if ok.sum() >= 5:
        log_ratio = np.log(actual[ok] * E_model[ok].sum() / (E_model[ok] * actual[ok].sum()))
        log_odds = np.log(p_i[ok] / (1 - p_i[ok]))
        np.savez(f"mnar_scatter_{name.split('_')[0]}.npz", log_ratio=log_ratio,
                 log_odds=log_odds, n_missing=actual[ok])
        c = np.corrcoef(log_ratio, log_odds - log_odds.mean())[0, 1]
        out["zonas_no_teste_MNAR"] = int(ok.sum())
        out["corr_desvio_vs_teoria_MNAR"] = round(float(c), 3)
        # quanto da distancia TV o ajuste MNAR-teorico elimina?
        E_adj = E_model * (p_i / (1 - p_i)) / np.maximum(
            (E_model * (p_i / (1 - p_i))).sum() / E_model.sum(), 1e-12)
        out["TV_modelo_ajustado_MNAR"] = tv(E_adj)
    return out

if __name__ == "__main__":
    results = []
    df = prepare(load_chicago())
    results.append(test_allocation(df, "chicago_CA77"))
    dfn = prepare(load_nyc())
    dfn = dfn[(dfn.ts >= "2013-01-01") & (dfn.ts <= "2019-12-31")]
    results.append(test_allocation(dfn, "nyc_boroughs_2013-19"))
    for r in results:
        print(json.dumps(r, indent=1, ensure_ascii=False))
    json.dump(results, open("resultados_validacao_natural.json", "w",
              encoding="utf-8"), indent=1, ensure_ascii=False)
