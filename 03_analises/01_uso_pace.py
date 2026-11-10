"""Uso (usage rate) x ritmo do time (pace) x lesões, por jogador-temporada (temporada regular, 2006-07 a 2025-26)."""
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.formula.api as smf

j = pd.read_pickle('dados/jogadores_20t_lesoes.pkl'); t = pd.read_pickle('dados/times_20t_lesoes.pkl')
E, M = pd.read_pickle('lesoes/episodios.pkl')
LES = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico']

r = j[(j.gameType == 'Regular Season') & j.usagePercentage.notna()].copy()
r['u_min'] = r.usagePercentage * r.numMinutes
ps = r.groupby(['personId', 'season']).agg(jogador=('playerKey', 'first'), jogos=('gameId', 'nunique'), minutos=('numMinutes', 'sum'),
                                           u_min=('u_min', 'sum'), uso_simples=('usagePercentage', 'mean'))
ps['uso'] = ps.u_min / ps.minutos; ps['mpj'] = ps.minutos / ps.jogos
tp = r.groupby(['personId', 'season', 'playerteamName']).size().reset_index(name='n').sort_values('n').drop_duplicates(['personId', 'season'], keep='last')
ps = ps.reset_index().merge(tp[['personId', 'season', 'playerteamName']], on=['personId', 'season'])
tr = t[t.gameType == 'Regular Season']
pace = tr.groupby(['season', 'teamName']).pace.mean().rename('pace').reset_index()
ps = ps.merge(pace, left_on=['season', 'playerteamName'], right_on=['season', 'teamName']).drop(columns='teamName')
jogos_temp = tr.groupby(['season', 'teamName']).gameId.nunique().groupby('season').max().rename('jogosTemporada')
ps = ps.merge(jogos_temp, on='season')

# lesões: jogos de temporada regular perdidos por lesão e afastamentos iniciados na temporada
ML = M[(M.categoria.isin(LES)) & (M.gameType == 'Regular Season')]
perd = ML.groupby(['personId', 'temporadaJogo']).gameId.nunique().rename('jogosPerdidosLesao').reset_index().rename(columns={'temporadaJogo': 'season'})
EL = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0)]
nles = EL.groupby(['personId', 'temporada']).size().rename('lesoes').reset_index().rename(columns={'temporada': 'season'})
ps = ps.merge(perd, on=['personId', 'season'], how='left').merge(nles, on=['personId', 'season'], how='left').fillna({'jogosPerdidosLesao': 0, 'lesoes': 0})
ps['taxaPerdida'] = ps.jogosPerdidosLesao / ps.jogosTemporada
ps['teveLesao'] = (ps.lesoes > 0).astype(int)
# próxima temporada (para olhar efeito defasado)
ps = ps.sort_values(['personId', 'season'])
temps = sorted(ps.season.unique()); nxt = {a: b for a, b in zip(temps, temps[1:])}
prox = ps[['personId', 'season', 'taxaPerdida']].rename(columns={'season': 'sp', 'taxaPerdida': 'taxaPerdidaProx'})
ps['sp'] = ps.season.map(nxt); ps = ps.merge(prox, on=['personId', 'sp'], how='left').drop(columns='sp')

base = ps[(ps.uso > 0.20) & (ps.jogos >= 20)].copy()
# padroniza dentro da temporada (o pace da liga subiu ~10 posses e o registro de lesões mudou ao longo dos anos)
for c in ['uso', 'pace', 'taxaPerdida', 'mpj']:
    base[c + '_z'] = base.groupby('season')[c].transform(lambda s: (s - s.mean()) / s.std())
print('jogador-temporadas:', len(base), '| jogadores:', base.personId.nunique())
print(base[['uso', 'pace', 'mpj', 'jogos', 'lesoes', 'jogosPerdidosLesao', 'taxaPerdida', 'teveLesao']].describe().round(3).T[['mean', '50%', 'min', 'max']].to_string())

def sp(a, b, d=base):
    m = d[[a, b]].dropna(); rho, p = stats.spearmanr(m[a], m[b]); return round(rho, 3), p, len(m)
print('\nSpearman (bruto)')
for a, b in [('uso', 'taxaPerdida'), ('pace', 'taxaPerdida'), ('uso', 'pace'), ('uso', 'lesoes'), ('pace', 'lesoes'), ('mpj', 'taxaPerdida'), ('uso', 'taxaPerdidaProx'), ('pace', 'taxaPerdidaProx')]:
    print(a, 'x', b, sp(a, b))
print('\nSpearman dentro da temporada (z)')
for a, b in [('uso_z', 'taxaPerdida_z'), ('pace_z', 'taxaPerdida_z'), ('uso_z', 'pace_z')]:
    print(a, 'x', b, sp(a, b))

# faixas de uso e de pace
base['faixaUso'] = pd.cut(base.uso, [0.20, 0.24, 0.28, 0.32, 1], labels=['20-24%', '24-28%', '28-32%', '32%+'])
base['faixaPace'] = base.groupby('season').pace.transform(lambda s: pd.qcut(s.rank(method='first'), 3, labels=['Lento', 'Médio', 'Rápido']))
f1 = base.groupby('faixaUso', observed=True).agg(n=('personId', 'size'), teveLesao=('teveLesao', 'mean'), jogosPerdidos=('jogosPerdidosLesao', 'mean'), taxa=('taxaPerdida', 'mean'), mpj=('mpj', 'mean'))
f2 = base.groupby('faixaPace', observed=True).agg(n=('personId', 'size'), teveLesao=('teveLesao', 'mean'), jogosPerdidos=('jogosPerdidosLesao', 'mean'), taxa=('taxaPerdida', 'mean'), uso=('uso', 'mean'))
f3 = base.pivot_table(index='faixaUso', columns='faixaPace', values='taxaPerdida', aggfunc='mean', observed=True)
print('\n', f1.round(3).to_string()); print('\n', f2.round(3).to_string()); print('\n% de jogos perdidos por lesão (uso x pace)\n', (f3 * 100).round(1).to_string())

# regressões: taxa de jogos perdidos ~ uso + pace + minutos + interação, com efeito fixo de temporada
base['uso100'] = base.uso * 100; base['taxa100'] = base.taxaPerdida * 100
for f in ['taxa100 ~ uso100 + pace + mpj + C(season)', 'taxa100 ~ uso100 * pace + mpj + C(season)']:
    m = smf.ols(f, data=base).fit(cov_type='cluster', cov_kwds={'groups': base.personId})
    keep = [k for k in m.params.index if not k.startswith('C(')]
    print('\n', f, '| R2', round(m.rsquared, 3)); print(pd.DataFrame({'coef': m.params[keep], 'p': m.pvalues[keep]}).round(4).to_string())
lg = smf.logit('teveLesao ~ uso100 + pace + mpj + C(season)', data=base).fit(disp=0, cov_type='cluster', cov_kwds={'groups': base.personId})
keep = ['uso100', 'pace', 'mpj']; print('\nlogit teveLesao: odds ratio'); print(pd.DataFrame({'OR': np.exp(lg.params[keep]), 'p': lg.pvalues[keep]}).round(4).to_string())
nx = base.dropna(subset=['taxaPerdidaProx']).copy(); nx['prox100'] = nx.taxaPerdidaProx * 100
m = smf.ols('prox100 ~ uso100 + pace + mpj + taxa100 + C(season)', data=nx).fit(cov_type='cluster', cov_kwds={'groups': nx.personId})
keep = ['uso100', 'pace', 'mpj', 'taxa100']; print('\nPróxima temporada | n', len(nx)); print(pd.DataFrame({'coef': m.params[keep], 'p': m.pvalues[keep]}).round(4).to_string())
pd.to_pickle(base, 'lesoes/uso_pace_base.pkl')
