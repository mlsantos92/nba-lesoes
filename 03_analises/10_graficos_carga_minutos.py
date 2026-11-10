"""Gráficos: minutos acima de 1 e 2 desvios-padrão da média do jogador e a incidência de lesões.

Entrada: lesoes/jogo_lesao.pkl (gerado por 07_jogo_da_lesao_base.py), uma linha por jogador-jogo.
Saída:   04_resultados/graficos/*.png e 04_resultados/carga_minutos.csv
Uso:     python 03_analises/10_graficos_carga_minutos.py <pasta com lesoes/> <pasta de saída 04_resultados>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("04_resultados")
(OUT / "graficos").mkdir(parents=True, exist_ok=True)
AZUL, LARANJA, CINZA, TINTA = "#2F5D9E", "#D2601A", "#9AA3AD", "#15202B"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": CINZA, "axes.labelcolor": TINTA, "xtick.color": TINTA, "ytick.color": TINTA})

r = pd.read_pickle(BASE / "lesoes" / "jogo_lesao.pkl")
r = r[r.zMax7d.notna()].copy()   # precisa de pelo menos um jogo nos 7 dias anteriores
# faixas exclusivas pelo jogo mais pesado (em minutos) dos 7 dias anteriores, em DP da média do jogador na temporada
r["faixa"] = pd.cut(r.zMax7d, [-np.inf, 1, 2, np.inf], labels=["Até 1 DP", "Entre 1 e 2 DP", "Acima de 2 DP"])
CATS = ["Ligamento", "Muscular", "Fratura", "Contusão (batidas)"]


def taxa_ic(s):
    """Lesões por 1.000 jogos com intervalo de 95% (aproximação de Poisson)."""
    n, k = len(s), s.sum()
    t = k / n * 1000
    e = 1.96 * np.sqrt(k) / n * 1000
    return pd.Series({"jogos": n, "lesoes": int(k), "por_1000": t, "ic_inf": t - e, "ic_sup": t + e})


tab = r.groupby("faixa", observed=True).lesao.apply(taxa_ic).unstack()
porcat = pd.DataFrame({c: r.groupby("faixa", observed=True).cat.apply(lambda s, c=c: (s == c).mean() * 1000) for c in CATS})

# logit condicional: compara cada jogo com os outros jogos do mesmo jogador na mesma temporada
r["z1a2"] = (r.faixa == "Entre 1 e 2 DP").astype(int)
r["zAcima2"] = (r.faixa == "Acima de 2 DP").astype(int)
X = ["z1a2", "zAcima2", "jogos7d", "b2b", "casa", "prorrogacao"]


def cl(y):
    d = r.dropna(subset=X)
    d = d[d.groupby("grupo")[y].transform("max") > 0]
    m = ConditionalLogit(d[y], d[X].astype(float), groups=d.grupo).fit(disp=0)
    ci = m.conf_int()
    return pd.DataFrame({"OR": np.exp(m.params), "ic_inf": np.exp(ci[0]), "ic_sup": np.exp(ci[1]), "p": m.pvalues}).loc[["z1a2", "zAcima2"]]


mods = {"Todas as lesões": cl("lesao")}
for c in CATS:
    r["y"] = (r.cat == c).astype(int)
    mods[c] = cl("y")
OR = pd.concat(mods, names=["lesao", "faixa"]).reset_index()
OR["faixa"] = OR.faixa.map({"z1a2": "Entre 1 e 2 DP", "zAcima2": "Acima de 2 DP"})

# ---------- gráfico 1: aumento do risco dentro do mesmo jogador, por papel no elenco ----------
r["papel"] = pd.qcut(r["mean"], 3, labels=["Reserva", "Rotação", "Titular"])
grupos = {"Todos": r}
grupos.update({str(k): d for k, d in r.groupby("papel", observed=True)})
lin = []
for nome, d in grupos.items():
    d = d.dropna(subset=X)
    d = d[d.groupby("grupo").lesao.transform("max") > 0]
    m = ConditionalLogit(d.lesao, d[X].astype(float), groups=d.grupo).fit(disp=0)
    ci = np.exp(m.conf_int())
    for k, fx in [("z1a2", "Entre 1 e 2 DP"), ("zAcima2", "Acima de 2 DP")]:
        lin.append(dict(grupo=nome, faixa=fx, min_medio=d["mean"].mean(), OR=np.exp(m.params[k]), ic_inf=ci.loc[k, 0], ic_sup=ci.loc[k, 1], p=m.pvalues[k]))
PAP = pd.DataFrame(lin)
fig, ax = plt.subplots(figsize=(7.6, 4.4))
nomes = list(grupos)
x = np.arange(len(nomes))
for desloc, faixa, cor in [(-.18, "Entre 1 e 2 DP", AZUL), (.18, "Acima de 2 DP", LARANJA)]:
    s = PAP[PAP.faixa == faixa].set_index("grupo").loc[nomes]
    v = (s.OR - 1) * 100
    ax.bar(x + desloc, v, width=.34, color=cor, label=faixa, zorder=2)
    ax.errorbar(x + desloc, v, yerr=[v - (s.ic_inf - 1) * 100, (s.ic_sup - 1) * 100 - v], fmt="none", ecolor=TINTA, capsize=3, lw=1, zorder=3)
    for xi, vi, hi, pv in zip(x + desloc, v, (s.ic_sup - 1) * 100, s.p):
        ax.text(xi, hi + 1.5, f"{vi:+.0f}%" + ("" if pv < .05 else "*"), ha="center", fontsize=9, color=TINTA)
mm = PAP.drop_duplicates("grupo").set_index("grupo").loc[nomes, "min_medio"]
ax.set_xticks(x, [f"{n}\n({m:.0f} min/jogo)" for n, m in zip(nomes, mm)])
ax.axhline(0, color=TINTA, lw=1)
ax.set_ylabel("Aumento do risco de lesão (%)")
ax.set_ylim(-15, max((PAP.ic_sup - 1) * 100) + 12)
ax.grid(axis="y", alpha=.25, zorder=0)
ax.legend(frameon=False, loc="upper right", title="Jogo mais pesado dos 7 dias anteriores", title_fontsize=8, fontsize=8)
ax.set_title("Semana com jogo acima da média de minutos: quanto sobe o risco", loc="left", fontsize=12, color=TINTA)
fig.text(.01, .01, "Mesmo jogador e temporada (logit condicional), contra semanas sem jogo acima de 1 DP. Barras: IC 95%. * sem significância (p ≥ 0,05).", fontsize=7.5, color="#566270")
fig.tight_layout(rect=(0, .04, 1, 1))
fig.savefig(OUT / "graficos" / "carga_risco_por_papel.png")

# ---------- gráfico 2: odds ratios do logit condicional ----------
ordem = ["Todas as lesões", "Ligamento", "Contusão (batidas)", "Muscular", "Fratura"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
y0 = np.arange(len(ordem))[::-1]
for desloc, faixa, cor in [(.14, "Entre 1 e 2 DP", AZUL), (-.14, "Acima de 2 DP", LARANJA)]:
    s = OR[OR.faixa == faixa].set_index("lesao").loc[ordem]
    ax.errorbar(s.OR, y0 + desloc, xerr=[s.OR - s.ic_inf, s.ic_sup - s.OR], fmt="o", color=cor, ms=6, capsize=3, lw=1.4, label=faixa)
    for yy, v, hi in zip(y0 + desloc, s.OR, s.ic_sup):
        ax.text(min(hi, 2.85) + .03, yy, f"{v:.2f}".replace(".", ","), va="center", fontsize=8, color=TINTA)
ax.axvline(1, color=TINTA, lw=1, ls="--")
ax.set_yticks(y0, ordem)
ax.set_xlim(.6, 3.0)
ax.set_xlabel("Odds ratio vs. semana sem jogo acima de 1 DP (1 = sem diferença)")
ax.grid(axis="x", alpha=.25)
ax.legend(frameon=False, loc="upper right", title="Jogo mais pesado da semana", title_fontsize=8, fontsize=8)
ax.set_title("Semana pesada e risco de lesão, por tipo", loc="left", fontsize=12, color=TINTA)
fig.text(.01, .01, "Logit condicional por jogador-temporada, controlando jogos na semana, jogos seguidos, mando e prorrogação. Barras: IC 95%.", fontsize=7.5, color="#566270")
fig.tight_layout(rect=(0, .04, 1, 1))
fig.savefig(OUT / "graficos" / "carga_odds_ratio.png")

# ---------- gráfico 3: gradiente pelo jogo imediatamente anterior ----------
r2 = pd.read_pickle(BASE / "lesoes" / "jogo_lesao.pkl").dropna(subset=["zMinAnterior"])
r2["fx"] = pd.cut(r2.zMinAnterior, [-np.inf, -1, 0, 1, 2, np.inf], labels=["< −1 DP", "−1 a 0", "0 a +1", "+1 a +2", "> +2 DP"])
g = r2.groupby("fx", observed=True).lesao.apply(taxa_ic).unstack()
fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(len(g))
ax.bar(x, g.por_1000, color=[CINZA, CINZA, CINZA, AZUL, LARANJA], width=.58, zorder=2)
ax.errorbar(x, g.por_1000, yerr=[g.por_1000 - g.ic_inf, g.ic_sup - g.por_1000], fmt="none", ecolor=TINTA, capsize=4, lw=1, zorder=3)
for i, t in enumerate(g.por_1000):
    ax.text(i, g.ic_sup.iloc[i] + .4, f"{t:.1f}".replace(".", ","), ha="center", fontsize=10, color=TINTA)
ax.set_xticks(x, g.index)
ax.set_ylabel("Lesões por 1.000 jogos")
ax.set_ylim(0, g.ic_sup.max() * 1.22)
ax.grid(axis="y", alpha=.25, zorder=0)
ax.set_title("Lesões conforme os minutos do jogo anterior", loc="left", fontsize=12, color=TINTA)
fig.text(.01, .01, "Minutos do jogo anterior em DP da média do jogador na temporada. Barras: IC 95%.", fontsize=7.5, color="#566270")
fig.tight_layout(rect=(0, .04, 1, 1))
fig.savefig(OUT / "graficos" / "carga_jogo_anterior.png")

saida = pd.concat({"incidencia": tab, **{f"porcategoria_{c}": porcat[c].to_frame("por_1000") for c in CATS}})
saida.to_csv(OUT / "carga_minutos.csv", float_format="%.4f")
OR.to_csv(OUT / "carga_minutos_odds_ratio.csv", index=False, float_format="%.4f")
PAP.to_csv(OUT / "carga_minutos_por_papel.csv", index=False, float_format="%.4f")
print(tab.round(2).to_string())
print(porcat.round(2).to_string())
print(OR.round(3).to_string())
print(PAP.round(3).to_string())
