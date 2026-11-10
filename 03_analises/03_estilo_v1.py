"""Estilo de jogo (agrupamento por estatísticas) x tipo e local de lesão."""
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from scipy import stats
import statsmodels.formula.api as smf

j = pd.read_pickle('dados/jogadores_20t_lesoes.pkl'); E, M = pd.read_pickle('lesoes/episodios.pkl')
LES = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico']
r = j[j.gameType == 'Regular Season']
c = {'fga': 'fieldGoalsAttempted', 'tpa': 'threePointersAttempted', 'fta': 'freeThrowsAttempted', 'orb': 'reboundsOffensive', 'drb': 'reboundsDefensive',
     'ast': 'assists', 'stl': 'steals', 'blk': 'blocks', 'pf': 'foulsPersonal', 'tov': 'turnovers', 'pts': 'points', 'min': 'numMinutes'}
ps = r.groupby(['personId', 'season']).agg(jogador=('playerKey', 'first'), jogos=('gameId', 'nunique'), **{k: (v, 'sum') for k, v in c.items()}).reset_index()
r2 = r.assign(u=r.usagePercentage * r.numMinutes).groupby(['personId', 'season']).u.sum()
ps = ps.merge(r2.rename('u'), on=['personId', 'season']); ps['uso'] = ps.u / ps['min']
ps = ps[(ps.jogos >= 20) & (ps['min'] / ps.jogos >= 15)].copy()
p36 = lambda x: x / ps['min'] * 36
F = pd.DataFrame({'tres_fga': ps.tpa / ps.fga, 'll_fga': ps.fta / ps.fga, 'orb36': p36(ps.orb), 'drb36': p36(ps.drb), 'ast36': p36(ps.ast),
                  'stl36': p36(ps.stl), 'blk36': p36(ps.blk), 'fga36': p36(ps.fga), 'uso': ps.uso}, index=ps.index)
Z = F.groupby(ps.season).transform(lambda s: (s - s.mean()) / s.std())  # relativo à temporada (o 3 explodiu no período)
km = KMeans(n_clusters=5, n_init=20, random_state=0).fit(Z.values)
ps['grupo'] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=Z.columns).round(2)
print(cent.to_string())
# nomeia pelos centróides
def nome(row):
    if row.blk36 > 0.8 and row.orb36 > 0.8: return 'Pivô / protetor de aro'
    if row.orb36 > 0.5 and row.tres_fga < -0.3: return 'Ala-pivô físico (garrafão)'
    if row.ast36 > 0.8: return 'Armador criador'
    if row.tres_fga > 0.5 and row.ll_fga < 0: return 'Arremessador de 3'
    if row.uso > 0.5 or row.ll_fga > 0.4: return 'Pontuador que infiltra'
    return 'Ala versátil'
nomes = {i: nome(cent.loc[i]) for i in cent.index}
# rótulos finais pelos centróides desta rodada (random_state fixo)
NOMES = {'Pivô / protetor de aro': 'Pivô / protetor de aro', 'Ala versátil': 'Defensor de perímetro', 'Armador criador': 'Armador pontuador (alto uso)',
         'Arremessador de 3': 'Arremessador de 3', 'Ala-pivô físico (garrafão)': 'Interno pontuador (garrafão)'}
nomes = {k: NOMES.get(v, v) for k, v in nomes.items()}
print(nomes)
ps['estilo'] = ps.grupo.map(nomes)
ps = ps.drop(columns='uso').join(F)

# lesões por jogador-temporada, por categoria e por parte do corpo
L = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0)]
cat = L.pivot_table(index=['personId', 'temporada'], columns='categoria', values='afastamentoId', aggfunc='count', fill_value=0)
top = L.parteDoCorpo.value_counts().head(10).index
prt = L[L.parteDoCorpo.isin(top)].pivot_table(index=['personId', 'temporada'], columns='parteDoCorpo', values='afastamentoId', aggfunc='count', fill_value=0)
ps = ps.merge(cat, left_on=['personId', 'season'], right_index=True, how='left').merge(prt, left_on=['personId', 'season'], right_index=True, how='left')
cols_c = [x for x in LES if x in ps.columns]; cols_p = list(top)
ps[cols_c + cols_p] = ps[cols_c + cols_p].fillna(0)
ps['min1000'] = ps['min'] / 1000
print('\njogador-temporadas:', len(ps), '| jogadores:', ps.personId.nunique())
print(ps.estilo.value_counts().to_string())

# taxa: lesões por 1.000 minutos jogados, por estilo
def taxa(cols):
    g = ps.groupby('estilo')[cols + ['min1000']].sum()
    return g[cols].div(g.min1000, axis=0)
tc = taxa(cols_c); tp = taxa(cols_p)
geral_c = ps[cols_c].sum() / ps.min1000.sum(); geral_p = ps[cols_p].sum() / ps.min1000.sum()
print('\nLesões por 1.000 min (categoria)\n', tc.round(3).to_string()); print('\nÍndice vs média (1 = média)\n', (tc / geral_c).round(2).to_string())
print('\nÍndice vs média por parte do corpo\n', (tp / geral_p).round(2).to_string())
# composição: % das lesões de cada estilo por categoria
comp = ps.groupby('estilo')[cols_c].sum(); comp = comp.div(comp.sum(axis=1), axis=0)
print('\nComposição das lesões de cada estilo (%)\n', (comp * 100).round(1).to_string())
chi = stats.chi2_contingency(ps.groupby('estilo')[cols_c].sum().values); print('qui-quadrado categoria x estilo p =', chi[1])
chi = stats.chi2_contingency(ps.groupby('estilo')[cols_p].sum().values); print('qui-quadrado parte x estilo p =', chi[1])

# Poisson com exposição em minutos, controlando temporada: efeito de cada característica (z) sobre cada categoria/parte
res = []
for alvo in cols_c + cols_p:
    d = ps.join(Z.add_suffix('_z')).rename(columns={alvo: 'y'})
    try:
        m = smf.poisson('y ~ tres_fga_z + ll_fga_z + orb36_z + drb36_z + ast36_z + stl36_z + blk36_z + uso_z + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
        for k in ['tres_fga_z', 'll_fga_z', 'orb36_z', 'drb36_z', 'ast36_z', 'stl36_z', 'blk36_z', 'uso_z']:
            res.append((alvo, k, np.exp(m.params[k]), m.pvalues[k]))
    except Exception as e:
        print('falhou', alvo, e)
R = pd.DataFrame(res, columns=['lesao', 'caracteristica', 'razao', 'p'])
sig = R[R.p < 0.01].sort_values('razao')
print('\nEfeitos significativos (p<0,01): razão de taxas por +1 desvio-padrão\n', sig.round(3).to_string(index=False))
pd.to_pickle((ps, cent, nomes, tc, tp, geral_c, geral_p, comp, R), 'lesoes/estilo.pkl')
