"""Estilo x uso x pace x lesões; comparação dentro da mesma posição e dentro do mesmo histórico de lesões."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf

ps, cent, nomes, *_ = pd.read_pickle('lesoes/estilo.pkl')
j = pd.read_pickle('dados/jogadores_20t_lesoes.pkl'); t = pd.read_pickle('dados/times_20t_lesoes.pkl')
E, M = pd.read_pickle('lesoes/episodios.pkl')
LES = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico']
temps = sorted(t.season.unique()); ti = {s: i for i, s in enumerate(temps)}

# posição: G/F/C mais frequente na carreira
pos = j[j.startingPosition.isin(['G', 'F', 'C'])].groupby('personId').startingPosition.agg(lambda s: s.value_counts().idxmax())
ps['posicao'] = ps.personId.map(pos).map({'G': 'Armador (G)', 'F': 'Ala (F)', 'C': 'Pivô (C)'})
# pace do time principal na temporada (z dentro da temporada) e uso (z)
r = j[j.gameType == 'Regular Season']
tm = r.groupby(['personId', 'season', 'playerteamName']).size().reset_index(name='n').sort_values('n').drop_duplicates(['personId', 'season'], keep='last')
pace = t[t.gameType == 'Regular Season'].groupby(['season', 'teamName']).pace.mean().rename('pace').reset_index()
ps = ps.merge(tm[['personId', 'season', 'playerteamName']], on=['personId', 'season']).merge(pace, left_on=['season', 'playerteamName'], right_on=['season', 'teamName'])
for c in ['pace', 'uso']:
    ps[c + '_z'] = ps.groupby('season')[c].transform(lambda s: (s - s.mean()) / s.std())
ps['mpj'] = ps['min'] / ps.jogos
# desfechos: lesões (afastamentos com jogo perdido) e jogos perdidos por lesão na temporada regular
L = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0)]
ps['lesoes'] = ps[[c for c in LES if c in ps.columns]].sum(axis=1)
ML = M[M.categoria.isin(LES) & (M.gameType == 'Regular Season')]
perd = ML.groupby(['personId', 'temporadaJogo']).gameId.nunique().rename('perdidos')
ps['perdidos'] = [perd.get((p, s), 0) for p, s in zip(ps.personId, ps.season)]
# histórico: lesões nas 3 temporadas anteriores
cnt = L.assign(ti=L.temporada.map(ti)).groupby(['personId', 'ti']).size()
ps['ti'] = ps.season.map(ti)
ps['hist3'] = [sum(cnt.get((p, k), 0) for k in range(i - 3, i)) for p, i in zip(ps.personId, ps.ti)]
ps['faixaHist'] = pd.cut(ps.hist3, [-1, 0, 2, 100], labels=['Sem lesão (3 temp.)', '1-2 lesões', '3+ lesões'])
d = ps[(ps.ti >= 3) & ps.posicao.notna()].copy()
d['min1000'] = d['min'] / 1000
print('jogador-temporadas:', len(d), '| jogadores:', d.personId.nunique())
print(pd.crosstab(d.posicao, d.estilo).to_string())

def taxa(g):
    return pd.Series({'n': len(g), 'lesoes_1000min': g.lesoes.sum() / g.min1000.sum(), 'jogosPerdidos_med': g.perdidos.mean(),
                      'uso': g.uso.mean(), 'pace_z': g.pace_z.mean(), 'hist3': g.hist3.mean()})
print('\n== Mesma posição, estilos diferentes')
A = d.groupby(['posicao', 'estilo']).apply(taxa, include_groups=False)
A = A[A.n >= 80]; print(A.round(3).to_string())
print('\n== Mesmo histórico, estilos diferentes')
B = d.groupby(['faixaHist', 'estilo'], observed=True).apply(taxa, include_groups=False)
B = B[B.n >= 80]; print(B.round(3).to_string())
# categoria dentro da mesma posição (índice vs média da posição)
print('\n== Categoria por estilo dentro da posição (índice vs média da posição)')
cats = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)']
out = []
for p_, g in d.groupby('posicao'):
    base = g[cats].sum() / g.min1000.sum()
    for e_, h in g.groupby('estilo'):
        if len(h) < 80: continue
        out.append(pd.Series(((h[cats].sum() / h.min1000.sum()) / base).round(2), name=(p_, e_)))
print(pd.DataFrame(out).to_string())

# modelo único: Poisson de lesões com exposição em minutos
m = smf.poisson('lesoes ~ C(estilo, Treatment("Arremessador de 3")) + uso_z + pace_z + hist3 + C(posicao) + C(season)', data=d,
                offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
k = [x for x in m.params.index if not x.startswith('C(season')]
print('\n== Poisson: lesões por minuto (razão de taxas)'); print(pd.DataFrame({'RR': np.exp(m.params[k]), 'p': m.pvalues[k]}).round(3).to_string())
for cat in cats:
    mm = smf.poisson(f'Q("{cat}") ~ C(estilo, Treatment("Arremessador de 3")) + uso_z + pace_z + hist3 + C(posicao) + C(season)', data=d,
                     offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
    kk = [x for x in mm.params.index if 'estilo' in x or x in ('uso_z', 'pace_z', 'hist3')]
    print('\n--', cat); print(pd.DataFrame({'RR': np.exp(mm.params[kk]), 'p': mm.pvalues[kk]}).round(3).to_string())
# interação estilo x histórico: o histórico pesa igual em todos os estilos?
mi = smf.poisson('lesoes ~ C(estilo) * hist3 + uso_z + pace_z + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
ki = [x for x in mi.params.index if ':hist3' in x]; print('\n== Interação estilo x histórico'); print(pd.DataFrame({'RR': np.exp(mi.params[ki]), 'p': mi.pvalues[ki]}).round(3).to_string())
mu = smf.poisson('lesoes ~ C(estilo) * uso_z + pace_z + hist3 + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
ku = [x for x in mu.params.index if ':uso_z' in x]; print('\n== Interação estilo x uso'); print(pd.DataFrame({'RR': np.exp(mu.params[ku]), 'p': mu.pvalues[ku]}).round(3).to_string())
mp = smf.poisson('lesoes ~ C(estilo) * pace_z + uso_z + hist3 + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
kp = [x for x in mp.params.index if ':pace_z' in x]; print('\n== Interação estilo x pace'); print(pd.DataFrame({'RR': np.exp(mp.params[kp]), 'p': mp.pvalues[kp]}).round(3).to_string())
pd.to_pickle((d, A, B), 'lesoes/estilo_cruzado.pkl')
