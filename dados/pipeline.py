# -*- coding: utf-8 -*-
"""
Pipeline empirico unificado: estimacao de intensidades de Poisson espaco-
temporais com localizacao faltante (modelos de Guigues et al. 2026) em dados
publicos reais.

Para cada dataset produz:
  1. Perfil de missingness: p-hat global, por hora-da-semana (T=168), por tipo;
     testes de homogeneidade (deviance binomial) p constante vs p_t vs p_{c,t}.
  2. Teste MNAR: onde existe rotulo de zona (bairro/community area) mesmo sem
     coordenada, testa se a taxa de faltante depende da zona (viola a hipotese
     central do modelo, p independente da localizacao).
  3. Magnitude do vies: subestimacao da intensidade total ao descartar
     registros sem localizacao (global e pico por janela).
  4. Validacao fora-da-amostra (folds por mes): prever a intensidade TOTAL
     observada (M1+M0) por janela t do fold excluido usando (a) nao corrigido,
     (b) corrigido com p unico, (c) corrigido com p_t. MSE tipo E_ct do artigo.
  5. Regularizacao + EM vs PG no dado real: ajusta o modelo 2 penalizado
     (grade 10x10, T=168) com pesos {0, w_cv} e compara EM penalizado vs
     gradiente projetado (iteracoes/tempo/objetivo), e qualidade OOS da
     previsao por zona dos counts reportados M1.

Uso: python pipeline.py <dataset>   com dataset em {nyc, chicago, seattle}
     (imprime relatorio em texto; resultados tambem em resultados_<ds>.json)
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6
T_BINS = 168            # hora-da-semana
GRID = 10


# ------------------------------- loaders ------------------------------------
def load_nyc():
    df = pd.read_csv("nyc_collisions.csv", low_memory=False)
    ts = pd.to_datetime(
        df["crash_date"].str.slice(0, 10) + " " + df["crash_time"].fillna("00:00"),
        errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    # (0,0) e coordenadas fora da cidade = invalidas
    bad = (lat.abs() < 1e-6) | (lat < 40.4) | (lat > 41.0) | (lon < -74.3) | (lon > -73.6)
    lat, lon = lat.mask(bad), lon.mask(bad)
    inj = pd.to_numeric(df["number_of_persons_injured"], errors="coerce").fillna(0)
    kil = pd.to_numeric(df["number_of_persons_killed"], errors="coerce").fillna(0)
    typ = np.where(kil > 0, "fatal", np.where(inj > 0, "injury", "damage-only"))
    zone = df["borough"].where(df["borough"].notna() & (df["borough"] != ""))
    return pd.DataFrame(dict(ts=ts, lat=lat, lon=lon, type=typ, zone_known=zone))


def load_chicago():
    df = pd.read_csv("chicago_crimes.csv", low_memory=False)
    ts = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    bad = (lat < 41.6) | (lat > 42.05) | (lon < -87.95) | (lon > -87.5)
    lat, lon = lat.mask(bad), lon.mask(bad)
    top = df["primary_type"].value_counts().index[:2]
    typ = np.where(df["primary_type"].isin(top), df["primary_type"], "OTHER")
    zone = pd.to_numeric(df["community_area"], errors="coerce")
    return pd.DataFrame(dict(ts=ts, lat=lat, lon=lon, type=typ,
                             zone_known=zone.astype("Int64").astype(str)
                             .where(zone.notna())))


def load_seattle():
    df = pd.read_csv("seattle_fire911.csv", low_memory=False)
    ts = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    bad = (lat < 47.4) | (lat > 47.75) | (lon < -122.5) | (lon > -122.2)
    lat, lon = lat.mask(bad), lon.mask(bad)
    t = df["type"].fillna("")
    typ = np.where(t.str.contains("Aid|Medic", case=False), "EMS",
                   np.where(t.str.contains("Fire", case=False), "Fire", "Other"))
    return pd.DataFrame(dict(ts=ts, lat=lat, lon=lon, type=typ,
                             zone_known=pd.Series([None] * len(df))))


LOADERS = dict(nyc=load_nyc, chicago=load_chicago, seattle=load_seattle)


# ------------------------------ preparo -------------------------------------
def prepare(df):
    df = df.dropna(subset=["ts"]).copy()
    df["miss"] = df["lat"].isna() | df["lon"].isna()
    df["tow"] = df["ts"].dt.dayofweek * 24 + df["ts"].dt.hour   # hora-da-semana
    df["week"] = df["ts"].dt.to_period("W").astype(str)
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    return df


def binom_dev_test(k, n, df_extra):
    """Deviance de H0: p comum vs H1: p por grupo (k sucessos em n por grupo)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    ok = n > 0
    k, n = k[ok], n[ok]
    p0 = k.sum() / n.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        ll1 = np.where(k > 0, k * np.log(k / n), 0) + \
              np.where(n - k > 0, (n - k) * np.log(1 - k / n), 0)
        ll0 = k * np.log(p0) + (n - k) * np.log(1 - p0)
    dev = 2 * (ll1.sum() - ll0.sum())
    dof = ok.sum() - 1 if df_extra is None else df_extra
    return float(dev), int(dof), float(stats.chi2.sf(dev, dof))


# --------------------------- analises 1-4 -----------------------------------
def analyze(df, name):
    R = dict(dataset=name, n_rows=int(len(df)))
    R["periodo"] = [str(df.ts.min())[:10], str(df.ts.max())[:10]]
    R["semanas"] = df["week"].nunique()
    R["p_global"] = float(df["miss"].mean())

    # p por hora-da-semana e por tipo
    g = df.groupby("tow")["miss"].agg(["sum", "count"])
    R["p_por_hora_min"] = float((g["sum"] / g["count"]).min())
    R["p_por_hora_max"] = float((g["sum"] / g["count"]).max())
    dev, dof, pv = binom_dev_test(g["sum"], g["count"], None)
    R["teste_p_depende_hora"] = dict(deviance=round(dev, 1), gl=dof, p_valor=pv)

    gt = df.groupby("type")["miss"].agg(["sum", "count"])
    R["p_por_tipo"] = {t: round(float(r["sum"] / r["count"]), 4)
                       for t, r in gt.iterrows()}
    dev, dof, pv = binom_dev_test(gt["sum"], gt["count"], None)
    R["teste_p_depende_tipo"] = dict(deviance=round(dev, 1), gl=dof, p_valor=pv)

    # MNAR: p por zona rotulada (entre linhas com rotulo de zona)
    dfz = df[df["zone_known"].notna()]
    R["frac_com_rotulo_zona"] = float(len(dfz) / len(df))
    if len(dfz) > 0 and dfz["zone_known"].nunique() > 1:
        gz = dfz.groupby("zone_known")["miss"].agg(["sum", "count"])
        gz = gz[gz["count"] >= 200]
        pz = gz["sum"] / gz["count"]
        dev, dof, pv = binom_dev_test(gz["sum"], gz["count"], None)
        R["teste_MNAR_p_depende_zona"] = dict(
            n_zonas=int(len(gz)), p_min=round(float(pz.min()), 4),
            p_max=round(float(pz.max()), 4), deviance=round(dev, 1),
            gl=dof, p_valor=pv)
    else:
        R["teste_MNAR_p_depende_zona"] = "sem rotulo de zona neste dataset"

    # vies de descarte: fracao subestimada da intensidade total
    R["subestimacao_total_%"] = round(100 * R["p_global"], 2)
    ph = g["sum"] / g["count"]
    R["subestimacao_pico_%_hora"] = round(100 * float(ph.max()), 2)

    # validacao OOS por mes: prever intensidade TOTAL por hora-da-semana
    months = sorted(df["month"].unique())
    if len(months) >= 6:
        err = dict(nao_corrigido=[], corr_p_unico=[], corr_p_t=[])
        for m in months:
            tr, te = df[df["month"] != m], df[df["month"] == m]
            # exposicao correta: dias observados / 7 (semanas ISO parciais
            # inflariam o denominador e deflariam o alvo)
            wk_tr = tr["ts"].dt.date.nunique() / 7
            wk_te = te["ts"].dt.date.nunique() / 7
            if wk_tr < 4 or wk_te == 0:
                continue
            tot_tr = tr.groupby("tow").size().reindex(range(T_BINS), fill_value=0)
            m1_tr = (~tr["miss"]).groupby(tr["tow"]).sum().reindex(range(T_BINS), fill_value=0)
            obs_te = te.groupby("tow").size().reindex(range(T_BINS), fill_value=0) / wk_te
            lam_unc = m1_tr / wk_tr
            p_hat = tr["miss"].mean()
            lam_c1 = m1_tr / wk_tr / (1 - p_hat)
            with np.errstate(divide="ignore", invalid="ignore"):
                lam_ct = (tot_tr / wk_tr).where(m1_tr > 0, 0.0)  # corrigido p_t = eq.(6) agregada
            err["nao_corrigido"].append(((lam_unc - obs_te) ** 2).mean())
            err["corr_p_unico"].append(((lam_c1 - obs_te) ** 2).mean())
            err["corr_p_t"].append(((lam_ct - obs_te) ** 2).mean())
        R["OOS_MSE_intensidade_total_por_hora"] = {
            k: round(float(np.mean(v)), 3) for k, v in err.items()}
    return R


# ------------------ analise 5: modelo 2 penalizado + EM vs PG ---------------
def fit_penalized_real(df, name, weights=(0.0, 1e-4)):
    """Monta M1[i,t], M0[t] (grade GRIDxGRID, T=168, tipos agregados) e roda
    EM penalizado vs gradiente projetado; avalia OOS por zona (ultimos ~20%
    das semanas como teste, alvo = M1 reportado por (i,t) no teste)."""
    d = df.dropna(subset=["ts"]).copy()
    qlat = d["lat"].quantile([0.005, 0.995])
    qlon = d["lon"].quantile([0.005, 0.995])
    lat0, lat1 = qlat.iloc[0], qlat.iloc[1]
    lon0, lon1 = qlon.iloc[0], qlon.iloc[1]
    ok = (~d["miss"]) & d["lat"].between(lat0, lat1) & d["lon"].between(lon0, lon1)
    d["zi"] = np.floor((d["lat"] - lat0) / (lat1 - lat0 + 1e-9) * GRID).clip(0, GRID - 1)
    d["zj"] = np.floor((d["lon"] - lon0) / (lon1 - lon0 + 1e-9) * GRID).clip(0, GRID - 1)
    d["zone"] = (d["zi"] * GRID + d["zj"]).where(ok)

    weeks = sorted(d["week"].unique())
    n_test = max(4, len(weeks) // 5)
    wk_tr, wk_te = set(weeks[:-n_test]), set(weeks[-n_test:])
    tr, te = d[d["week"].isin(wk_tr)], d[d["week"].isin(wk_te)]
    Ntr, Nte = len(wk_tr), len(wk_te)
    I = GRID * GRID

    def counts(dd):
        m1 = np.zeros((I, T_BINS))
        rep = dd[dd["zone"].notna()]
        np.add.at(m1, (rep["zone"].astype(int).values, rep["tow"].values), 1)
        m0 = np.zeros(T_BINS)
        mis = dd[dd["zone"].isna()]
        np.add.at(m0, mis["tow"].values, 1)
        return m1, m0

    M1, M0 = counts(tr)
    M1te, _ = counts(te)
    D = 1.0  # janelas de 1h

    # penalizacao: vizinhanca 4-conexa + grupos de tempo (mesma hora, dias uteis/fds)
    edges = []
    for r in range(GRID):
        for c in range(GRID):
            k = r * GRID + c
            if r + 1 < GRID:
                edges.append((k, k + GRID))
            if c + 1 < GRID:
                edges.append((k, k + 1))
    edges = np.array(edges)
    groups = [[dday * 24 + h for dday in range(5)] for h in range(24)] + \
             [[dday * 24 + h for dday in (5, 6)] for h in range(24)]

    def make_obj(w):
        WN2 = w * Ntr * Ntr

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
            return float((Ntr * D * S - M0 * np.log(S)).sum()
                         - (M1 * np.log(lam)).sum() + pen_grad(lam)[0])

        def gr(lam):
            S = lam.sum(0)
            return (Ntr * D - M0 / S)[None, :] - M1 / lam + pen_grad(lam)[1]

        return f, gr, pen_grad

    from scipy.optimize import minimize
    results = {}
    for w in weights:
        f, gr, pen_grad = make_obj(w)
        lam0 = np.full((I, T_BINS), max(M1.sum() / (Ntr * I * T_BINS), 0.1))

        # PG-Armijo
        t0 = time.perf_counter()
        lam, fv, step, stall, it_pg = lam0.copy(), None, 1.0 / Ntr, 0, 0
        fv = f(lam)
        for it in range(1, 1501):
            g = gr(lam)
            tstep = step
            for _ in range(50):
                ln = np.maximum(lam - tstep * g, EPS)
                fn = f(ln)
                if fn <= fv + 1e-4 * (g * (ln - lam)).sum():
                    break
                tstep *= 0.5
            rel = (fv - fn) / max(abs(fv), 1.0)
            lam, fv = ln, fn
            step = tstep / 0.5
            stall = stall + 1 if rel < 1e-10 else 0
            it_pg = it
            if stall >= 6:
                break
        t_pg = time.perf_counter() - t0
        lam_pg, f_pg = lam, fv

        # EM penalizado (M-step L-BFGS-B)
        t0 = time.perf_counter()
        lam, fv, stall, it_em = lam0.copy(), f(lam0), 0, 0
        for it in range(1, 201):
            S = lam.sum(0)
            C = M1 + M0[None, :] * lam / S[None, :]

            def q(x):
                l = x.reshape(I, T_BINS)
                return (Ntr * D * l.sum(0)).sum() - (C * np.log(l)).sum() + pen_grad(l)[0]

            def gq(x):
                l = x.reshape(I, T_BINS)
                return ((Ntr * D) - C / l + pen_grad(l)[1]).ravel()

            res = minimize(q, lam.ravel(), jac=gq, method="L-BFGS-B",
                           bounds=[(EPS, None)] * (I * T_BINS),
                           options=dict(maxiter=150, ftol=1e-12))
            lam = res.x.reshape(I, T_BINS)
            fn = f(lam)
            rel = (fv - fn) / max(abs(fv), 1.0)
            fv, it_em = fn, it
            stall = stall + 1 if rel < 1e-10 else 0
            if stall >= 6:
                break
        t_em = time.perf_counter() - t0
        lam_em, f_em = lam, fv

        # OOS por zona: alvo = contagem reportada no teste; preditor = lam*(1-p_t)*Nte
        p_t = M0 / np.maximum(M0 + M1.sum(0), 1)
        pred = lam_em * (1 - p_t)[None, :] * Nte
        mse_zone = float(((pred - M1te) ** 2).mean())
        base = (M1 / Ntr * (1)) * Nte * 0 + M1 / Ntr * Nte  # nao regularizado, nao corrigido
        mse_base = float(((base - M1te) ** 2).mean())
        results[str(w)] = dict(
            PG=dict(iters=it_pg, tempo_s=round(t_pg, 1), f=round(f_pg, 2)),
            EM=dict(iters=it_em, tempo_s=round(t_em, 1), f=round(f_em, 2)),
            gap_EM_menos_PG=round(f_em - f_pg, 4),
            OOS_MSE_M1_por_zona_hora=round(mse_zone, 4),
            OOS_MSE_M1_baseline_empirico=round(mse_base, 4))
    return results


# --------------------------------- main -------------------------------------
if __name__ == "__main__":
    name = sys.argv[1]
    df = prepare(LOADERS[name]())
    R = analyze(df, name)
    print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
    R["modelo2_penalizado"] = fit_penalized_real(df, name)
    print(json.dumps({"modelo2_penalizado": R["modelo2_penalizado"]},
                     indent=1, ensure_ascii=False))
    with open(f"resultados_{name}.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False, default=str)
