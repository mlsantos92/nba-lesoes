"""Lesão no jogo x defesa física do adversário x carga de minutos fora do padrão nos 7 dias anteriores."""
import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

j = pd.read_pickle('dados/jogadores_20t_lesoes.pkl'); t = pd.read_pickle('dados/times_20t_lesoes.pkl')
E, M = pd.read_pickle('lesoes/episodios.pkl')
LES = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico']

# ---------- defesa física do adversário (média da temporada regular, z dentro da temporada)
tr = t[t.gameType == 'Regular Season'].copy()
tr['faltas100'] = tr.foulsPersonal / tr.possessions * 100
tr['tocos100'] = tr.blocks / tr.possessions * 100
tr['ftrCedido'] = tr.opponentFreeThrowAttemptRate
D = tr.groupby(['season', 'teamName'])[['faltas100', 'ftrCedido', 'tocos100']].mean()
Dz = D.groupby(level=0).transform(lambda s: (s - s.mean()) / s.std())
Dz['contato'] = Dz[['faltas100', 'ftrCedido', 'tocos100']].mean(axis=1)
Dz['contato'] = Dz.groupby(level=0).contato.transform(lambda s: (s - s.mean()) / s.std())
Dz.columns = ['adv_' + c for c in Dz.columns]
Dz = Dz.reset_index().rename(columns={'teamName': 'opponentteamName'})
# contato no próprio jogo: faltas cometidas pelo adversário por 100 posses
g = tr[['gameId', 'teamName', 'personalFoulsDrawn', 'possessions', 'numMinutes']].rename(columns={'teamName': 'playerteamName'})
g['faltasAdvJogo100'] = g.personalFoulsDrawn / g.possessions * 100
g['prorrogacao'] = (g.numMinutes > 241).astype(int)

# ---------- jogos do jogador (temporada regular, entrou em quadra)
r = j[(j.gameType == 'Regular Season') & (j.numMinutes > 0)].copy()
r['data'] = pd.to_datetime(r.gameDateTimeEst).dt.normalize()
r = r.merge(g[['gameId', 'playerteamName', 'faltasAdvJogo100', 'prorrogacao']], on=['gameId', 'playerteamName'], how='left')
r = r.merge(Dz, on=['season', 'opponentteamName'], how='left')
r = r.sort_values(['personId', 'data']).reset_index(drop=True)
st = r.groupby(['personId', 'season']).numMinutes.agg(['mean', 'std', 'size'])
r = r.join(st, on=['personId', 'season'])
r = r[(r['size'] >= 20) & (r['mean'] >= 15)].copy()     # mesmo recorte das análises de estilo
r['zMin'] = (r.numMinutes - r['mean']) / r['std']
r['faltasSofridas'] = r.foulsAgainst

# ---------- jogo da lesão: último jogo antes do início do afastamento, até 3 dias antes
L = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0)].copy()
L['inicio'] = pd.to_datetime(L.inicio).dt.normalize().astype('datetime64[ns]'); L = L[L.personId.notna()]; L['personId'] = L.personId.astype('int64')
cand = r[['personId', 'data', 'gameId']].astype({'personId': 'int64', 'data': 'datetime64[ns]'}).sort_values('data')
L = pd.merge_asof(L.sort_values('inicio'), cand.rename(columns={'data': 'dataJogo'}), left_on='inicio', right_on='dataJogo', by='personId',
                  direction='backward', allow_exact_matches=True)
L = L[(L.inicio - L.dataJogo).dt.days.between(0, 3)]
L = L.drop_duplicates(['personId', 'gameId'])
r['personId'] = r.personId.astype('int64'); r = r.merge(L[['personId', 'gameId', 'categoria', 'parteDoCorpo']], on=['personId', 'gameId'], how='left')
r['lesao'] = r.categoria.notna().astype(int)
print('afastamentos por lesão:', int((E.categoria.isin(LES) & (E.jogosPerdidos > 0)).sum()), '| ligados a um jogo (até 3 dias antes):', r.lesao.sum())

# ---------- carga recente: jogos nos 7 dias ANTERIORES (sem contar o jogo atual)
def janela(grp):
    d = grp.data.values.astype('datetime64[D]').astype(int); z = grp.zMin.values; mn = grp.numMinutes.values
    n = len(d); a1 = np.zeros(n, int); a2 = np.zeros(n, int); nj = np.zeros(n, int); m7 = np.zeros(n); b2b = np.zeros(n, int); mx = np.full(n, np.nan)
    lo = 0
    for i in range(n):
        while d[lo] < d[i] - 7: lo += 1
        w = slice(lo, i)
        nj[i] = i - lo; m7[i] = mn[w].sum()
        if i - lo: a1[i] = (z[w] > 1).any(); a2[i] = (z[w] > 2).any(); mx[i] = z[w].max()
        b2b[i] = i > 0 and d[i] - d[i - 1] == 1
    return pd.DataFrame({'pesado1dp_7d': a1, 'pesado2dp_7d': a2, 'jogos7d': nj, 'min7d': m7, 'b2b': b2b, 'zMax7d': mx}, index=grp.index)
r = r.join(pd.concat([janela(x) for _, x in r.groupby(['personId', 'season'])]))
r['zMinAnterior'] = r.groupby(['personId', 'season']).zMin.shift(1)
r['grupo'] = r.personId.astype(str) + '_' + r.season
r['casa'] = (r.home.astype(str) == 'Mandante').astype(int)
r['cat'] = r.categoria.fillna('')
pd.to_pickle(r[['personId', 'season', 'gameId', 'data', 'grupo', 'opponentteamName', 'numMinutes', 'zMin', 'zMinAnterior', 'zMax7d', 'pesado1dp_7d', 'pesado2dp_7d', 'jogos7d',
                'min7d', 'b2b', 'casa', 'prorrogacao', 'faltasAdvJogo100', 'faltasSofridas', 'adv_faltas100', 'adv_ftrCedido', 'adv_tocos100', 'adv_contato',
                'lesao', 'cat', 'parteDoCorpo', 'mean', 'startingPosition']], 'lesoes/jogo_lesao.pkl')
print('jogador-jogos:', len(r), '| taxa de lesão por 1.000 jogos:', round(r.lesao.mean() * 1000, 2))
