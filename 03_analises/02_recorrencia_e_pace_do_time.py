"""1) Recorrência de lesão no mesmo local; 2) uso controlado pelo histórico; 3) ritmo do time x lesões do elenco."""
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.formula.api as smf

j = pd.read_pickle('dados/jogadores_20t_lesoes.pkl'); t = pd.read_pickle('dados/times_20t_lesoes.pkl')
E, M = pd.read_pickle('lesoes/episodios.pkl')
LES = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico']
temps = sorted(t.season.unique()); ti = {s: i for i, s in enumerate(temps)}

# ---------- 1) recorrência por parte do corpo ----------
r = j[j.gameType == 'Regular Season']
ps = r.groupby(['personId', 'season']).agg(jogos=('gameId', 'nunique'), minutos=('numMinutes', 'sum')).reset_index()
ps = ps[ps.jogos >= 20]; ps['mpj'] = ps.minutos / ps.jogos; ps['ti'] = ps.season.map(ti)
L = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0) & (E.parteDoCorpo != '')].copy(); L['ti'] = L.temporada.map(ti)
partes = L.parteDoCorpo.value_counts(); partes = partes[partes >= 150].index.tolist()
hist = L.groupby(['personId', 'parteDoCorpo']).ti.apply(lambda s: np.sort(s.unique())).to_dict()
lado = L.groupby(['personId', 'parteDoCorpo', 'lado']).ti.apply(lambda s: np.sort(s.unique())).to_dict()
alvo = ps[ps.ti >= 3]  # 2009-10 em diante: todos têm 3 temporadas de histórico possível
rows = []
for p in partes:
    for x in alvo.itertuples():
        h = hist.get((x.personId, p), np.array([]))
        rows.append((x.personId, x.season, x.mpj, p, int(((h >= x.ti - 3) & (h < x.ti)).any()), int((h == x.ti).any())))
R = pd.DataFrame(rows, columns=['personId', 'season', 'mpj', 'parte', 'antes', 'agora'])
tab = R.groupby(['parte', 'antes']).agora.agg(['mean', 'size']).unstack()
tab.columns = ['risco_sem_hist', 'risco_com_hist', 'n_sem', 'n_com']
tab['risco_relativo'] = tab.risco_com_hist / tab.risco_sem_hist
tab = tab.sort_values('risco_relativo', ascending=False)
print('Risco de lesão na parte X na temporada, com e sem lesão na mesma parte nas 3 temporadas anteriores')
print((tab[['risco_sem_hist', 'risco_com_hist']] * 100).round(1).join(tab[['risco_relativo', 'n_com']].round(2)).to_string())
geral = R.groupby('antes').agora.mean(); print('Geral (todas as partes):', (geral * 100).round(2).to_dict(), 'RR', round(geral[1] / geral[0], 2))
lg = smf.logit('agora ~ antes + mpj + C(parte) + C(season)', data=R).fit(disp=0, cov_type='cluster', cov_kwds={'groups': R.personId})
print('Logit (controla minutos, parte e temporada): odds ratio do histórico', round(np.exp(lg.params['antes']), 2), 'p', lg.pvalues['antes'])
# mesmo lado
ld = L[L.lado != '']
mesmo = diferente = 0
for (pid, p, lad), tis in lado.items():
    if lad == '' or p not in partes: continue
    outro = 'Direito' if lad == 'Esquerdo' else 'Esquerdo'
    for tt in tis:
        prev_m = lado.get((pid, p, lad), np.array([])); prev_o = lado.get((pid, p, outro), np.array([]))
        if ((prev_m < tt) & (prev_m >= tt - 3)).any(): mesmo += 1
        elif ((prev_o < tt) & (prev_o >= tt - 3)).any(): diferente += 1
print(f'Recorrências com lado informado: mesmo lado {mesmo}, lado oposto {diferente} ({mesmo / (mesmo + diferente):.0%} no mesmo lado)')
pd.to_pickle(tab, 'lesoes/recorrencia.pkl')

# ---------- 2) uso > 20% controlando histórico de lesões ----------
b = pd.read_pickle('lesoes/uso_pace_base.pkl')
Lall = L.groupby(['personId', 'ti']).size()
b['ti'] = b.season.map(ti)
b['lesoes3anos'] = [sum(Lall.get((p, k), 0) for k in range(i - 3, i)) for p, i in zip(b.personId, b.ti)]
b2 = b[b.ti >= 3].copy()
for f in ['taxa100 ~ uso100 + pace + mpj + C(season)', 'taxa100 ~ uso100 + pace + mpj + lesoes3anos + C(season)']:
    m = smf.ols(f, data=b2).fit(cov_type='cluster', cov_kwds={'groups': b2.personId})
    k = [x for x in m.params.index if not x.startswith('C(') and x != 'Intercept']
    print('\n', f, '| n', len(b2), '| R2', round(m.rsquared, 3)); print(pd.DataFrame({'coef': m.params[k], 'p': m.pvalues[k]}).round(4).to_string())
print('Spearman uso x lesões nas 3 temporadas anteriores:', stats.spearmanr(b2.uso, b2.lesoes3anos)[0].round(3))

# ---------- 3) ritmo do time x lesões do elenco ----------
tr = t[t.gameType == 'Regular Season']
T = tr.groupby(['season', 'teamName']).agg(pace=('pace', 'mean'), jogos=('gameId', 'nunique')).reset_index()
ML = M[M.categoria.isin(LES) & (M.gameType == 'Regular Season')]
g = ML.groupby(['temporadaJogo', 'time']).agg(jogosPerdidos=('gameId', 'size'), jogadores=('personId', 'nunique')).reset_index().rename(columns={'temporadaJogo': 'season', 'time': 'teamName'})
ne = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0)].groupby(['temporada', 'time']).size().rename('lesoes').reset_index().rename(columns={'temporada': 'season', 'time': 'teamName'})
T = T.merge(g, on=['season', 'teamName'], how='left').merge(ne, on=['season', 'teamName'], how='left').fillna(0)
T['perdidosPorJogo'] = T.jogosPerdidos / T.jogos; T['lesoesPor82'] = T.lesoes / T.jogos * 82
for c in ['pace', 'perdidosPorJogo', 'lesoesPor82']:
    T[c + '_z'] = T.groupby('season')[c].transform(lambda s: (s - s.mean()) / s.std())
print('\nTimes-temporada:', len(T))
for a, bb in [('pace', 'perdidosPorJogo'), ('pace', 'lesoesPor82'), ('pace_z', 'perdidosPorJogo_z'), ('pace_z', 'lesoesPor82_z')]:
    rho, p = stats.spearmanr(T[a], T[bb]); print(a, 'x', bb, round(rho, 3), round(p, 4))
T['terco'] = T.groupby('season').pace.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=['Lento', 'Médio', 'Rápido']))
print(T.groupby('terco', observed=True)[['pace', 'perdidosPorJogo', 'lesoesPor82']].mean().round(2).to_string())
for f in ['perdidosPorJogo ~ pace + C(season)', 'lesoesPor82 ~ pace + C(season)']:
    m = smf.ols(f, data=T).fit(cov_type='cluster', cov_kwds={'groups': T.teamName}); print(f, 'coef pace', round(m.params['pace'], 4), 'p', round(m.pvalues['pace'], 4))
# defasado: ritmo desta temporada -> lesões da próxima
T['ti'] = T.season.map(ti); nxt = T[['teamName', 'ti', 'perdidosPorJogo_z']].rename(columns={'perdidosPorJogo_z': 'prox'}); nxt['ti'] -= 1
T2 = T.merge(nxt, on=['teamName', 'ti']); print('pace_z -> perdidos próxima temporada (z):', round(stats.spearmanr(T2.pace_z, T2.prox)[0], 3), round(stats.spearmanr(T2.pace_z, T2.prox)[1], 4))
# por categoria
for c in ['Muscular', 'Ligamento', 'Fratura', 'Contusão (batidas)']:
    gc = M[(M.categoria == c) & (M.gameType == 'Regular Season')].groupby(['temporadaJogo', 'time']).size().rename('x').reset_index().rename(columns={'temporadaJogo': 'season', 'time': 'teamName'})
    Tc = T.merge(gc, on=['season', 'teamName'], how='left').fillna({'x': 0}); Tc['xz'] = Tc.groupby('season').x.transform(lambda s: (s - s.mean()) / s.std())
    rho, p = stats.spearmanr(Tc.pace_z, Tc.xz); print('categoria', c, round(rho, 3), round(p, 4))
pd.to_pickle(T, 'lesoes/time_pace.pkl')
