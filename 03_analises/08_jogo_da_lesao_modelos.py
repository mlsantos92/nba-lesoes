import pandas as pd, numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit
import statsmodels.formula.api as smf

r = pd.read_pickle('lesoes/jogo_lesao.pkl')
r = r.dropna(subset=['adv_contato']).copy()
r['faltasSofridas36'] = r.faltasSofridas / r.numMinutes * 36
cats = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)']
taxa = lambda s: round(s.mean() * 1000, 2)

print('== Taxa de lesão por 1.000 jogos, por quintil da defesa física do adversário (temporada)')
r['qContato'] = pd.qcut(r.adv_contato, 5, labels=['Q1 menos física', 'Q2', 'Q3', 'Q4', 'Q5 mais física'])
T1 = r.groupby('qContato', observed=True).agg(jogos=('lesao', 'size'), total=('lesao', taxa), **{c: ('cat', lambda s, c=c: round((s == c).mean() * 1000, 2)) for c in cats})
print(T1.to_string())
for c in ['adv_faltas100', 'adv_ftrCedido', 'adv_tocos100']:
    q = pd.qcut(r[c], 5, labels=False); print(c, 'Q1→Q5:', [taxa(r.lesao[q == i]) for i in range(5)])

print('\n== Carga: algum jogo nos 7 dias anteriores acima da média da temporada')
for f, nm in [('pesado1dp_7d', '>1 DP'), ('pesado2dp_7d', '>2 DP')]:
    a = r.groupby(f).agg(jogos=('lesao', 'size'), total=('lesao', taxa), **{c: ('cat', lambda s, c=c: round((s == c).mean() * 1000, 2)) for c in cats})
    print(nm); print(a.to_string())
r['faixaAnt'] = pd.cut(r.zMinAnterior, [-99, -1, 0, 1, 2, 99], labels=['<-1', '-1 a 0', '0 a 1', '1 a 2', '>2'])
print('\nJogo imediatamente anterior (z de minutos) →', r.groupby('faixaAnt', observed=True).lesao.agg(['size', taxa]).to_string())
print('\nJogos nos 7 dias anteriores →', r.groupby(r.jogos7d.clip(upper=4)).lesao.agg(['size', taxa]).to_string())
print('back-to-back →', r.groupby('b2b').lesao.agg(['size', taxa]).to_string())

# ---------- dentro do mesmo jogador-temporada (logit condicional): compara o jogo da lesão com os outros jogos do próprio jogador
def cl(y, X, nome):
    d = r.dropna(subset=X + [y]); d = d[d.groupby('grupo')[y].transform('max') > 0]
    m = ConditionalLogit(d[y], d[X], groups=d.grupo).fit(disp=0)
    out = pd.DataFrame({'OR': np.exp(m.params), 'ic95_inf': np.exp(m.conf_int()[0]), 'ic95_sup': np.exp(m.conf_int()[1]), 'p': m.pvalues}).round(3)
    print(f'\n== {nome} (eventos: {int(d[y].sum())})'); print(out.to_string()); return out
base = ['jogos7d', 'b2b', 'casa', 'prorrogacao']
o1 = cl('lesao', ['adv_contato', 'pesado1dp_7d'] + base, 'Lesão: contato do adversário + jogo >1 DP nos 7 dias')
o2 = cl('lesao', ['adv_contato', 'pesado2dp_7d'] + base, 'Lesão: contato do adversário + jogo >2 DP nos 7 dias')
o3 = cl('lesao', ['adv_faltas100', 'adv_ftrCedido', 'adv_tocos100', 'zMax7d'] + base, 'Componentes do contato + maior z de minutos nos 7 dias')
o4 = cl('lesao', ['faltasAdvJogo100', 'faltasSofridas36', 'adv_contato', 'pesado1dp_7d'] + base, 'Contato no próprio jogo (faltas do adversário, faltas sofridas por 36 min)')
res = []
for c in cats:
    r['y'] = (r.cat == c).astype(int)
    o = cl('y', ['adv_contato', 'pesado1dp_7d', 'pesado2dp_7d'] + base, f'Categoria: {c}')
    res.append(o.assign(categoria=c))
pd.to_pickle((r, T1, o1, o2, o3, o4, pd.concat(res)), 'lesoes/jogo_lesao_res.pkl')
