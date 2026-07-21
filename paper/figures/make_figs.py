# -*- coding: utf-8 -*-
"""Gera fig_convergence.pdf e fig_mnar.pdf para o paper (vetor, print-first).
Paleta categórica validada (dataviz): azul #2a78d6, verde #008300,
magenta #e87ba4 — rótulos diretos + estilos de linha distintos como
codificação secundária."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, GREEN, MAGENTA = "#2a78d6", "#008300", "#e87ba4"
INK, MUTED = "#333333", "#777777"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42,
})

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

# ------------------------- Figura 1: convergencia ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, w, wl in zip(axes, ["0.001", "0.01"], [r"$w=10^{-3}$", r"$w=10^{-2}$"]):
    d = np.load(f"../../em_experiment/hist_w{w}.npz")
    fmin = min(d["pg"][:, 2].min(), d["em"][:, 2].min(),
               d["gem"][:, 2].min(), float(d["fstar"]))
    series = [("PG (Armijo)", d["pg"], BLUE, "-"),
              ("EM", d["em"], GREEN, "--"),
              ("GEM", d["gem"], MAGENTA, "-.")]
    for name, h, col, ls in series:
        it, gap = h[:, 0], np.maximum(h[:, 2] - fmin, 1e-9)
        rel = gap / max(abs(fmin), 1.0)
        ax.plot(it + 1, rel, ls, color=col, linewidth=1.5, zorder=3)
        k = len(it) - 1
        dy = -11 if name == "GEM" else 4
        ax.annotate(name, (it[k] + 1, rel[k]), textcoords="offset points",
                    xytext=(4, dy), fontsize=8, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_title(wl, loc="left", fontsize=9)
    style_ax(ax)
axes[0].set_ylabel("relative objective gap")
fig.tight_layout()
fig.savefig("fig_convergence.pdf", bbox_inches="tight")
plt.close(fig)

# --------------------------- Figura 2: MNAR ---------------------------------
d = np.load("../../dados/mnar_scatter_chicago.npz")
x = d["log_odds"] - d["log_odds"].mean()
y = d["log_ratio"]
corr = np.corrcoef(y, x)[0, 1]
fig, ax = plt.subplots(figsize=(4.2, 3.4))
lim = [min(x.min(), y.min()) - 0.15, max(x.max(), y.max()) + 0.15]
ax.plot(lim, lim, color=MUTED, linewidth=0.9, linestyle=":", zorder=2)
ax.annotate("$y=x$", (lim[1] - 0.35, lim[1] - 0.22), fontsize=8, color=MUTED)
ax.scatter(x, y, s=22, facecolor=BLUE, edgecolor="white", linewidth=0.6,
           zorder=3)
ax.set_xlabel(r"MNAR signature $\log\{p_i/(1-p_i)\}$ (centered)")
ax.set_ylabel("log deviation, actual vs. model allocation")
ax.annotate(f"r = {corr:.3f}", (0.05, 0.92), xycoords="axes fraction",
            fontsize=9, color=INK)
ax.set_xlim(lim); ax.set_ylim(lim)
style_ax(ax)
ax.grid(True, axis="x", color="#dddddd", linewidth=0.5)
fig.tight_layout()
fig.savefig("fig_mnar.pdf", bbox_inches="tight")
plt.close(fig)
print("figuras ok; corr =", round(float(corr), 4))
