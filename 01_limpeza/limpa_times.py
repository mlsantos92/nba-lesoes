"""Limpeza da tabela de times (TeamStatisticsExtended), espelhando o tratamento dos jogadores."""
import pandas as pd, numpy as np

t = pd.read_pickle('dados/teams_raw.pkl').copy()
j = pd.read_pickle('dados/jogadores_final.pkl')
log = []  # (etapa, linhas, observação)
corr = []  # correções linha a linha: gameId, time, coluna, antes, depois, motivo

s = t.gameId.astype(str).str.zfill(8)
tipo_id = s.str[0]

# 1. fora: pré-temporada (1), All-Star (3), final da NBA Cup (6)
for cod, nome, obs in [('1', 'Pré-temporada removida', 'gameId começa com 1'),
                       ('3', 'All-Star removido', 'não é jogo oficial'),
                       ('6', 'Final da NBA Cup removida', 'a final não conta na classificação (igual aos jogadores)')]:
    n = int((tipo_id == cod).sum()); log.append((nome, f'{n} ({n // 2} jogos)', obs))
t = t[tipo_id.isin(['2', '4', '5'])].copy()
s = t.gameId.astype(str).str.zfill(8)

# 2. tipo de jogo vazio, pelo 1º dígito do gameId
vazio = t.gameType.isna()
t.loc[vazio, 'gameType'] = s[vazio].str[0].map({'2': 'Regular Season', '4': 'Playoffs', '5': 'Play-in Tournament'})
log.append(('Tipo de jogo vazio preenchido pelo gameId', f'{int(vazio.sum())}', '2 = temporada regular, 4 = playoffs, 5 = play-in'))

# 3. rótulos padronizados
PAD = {'East First Round': 'East - First Round', 'West First Round': 'West - First Round',
       'East Conf. Semifinals': 'East - Conf. Semifinals', 'West Conf. Semifinals': 'West - Conf. Semifinals',
       'East Conf. Finals': 'East - Conf. Finals', 'West Conf. Finals': 'West - Conf. Finals',
       'NBA Cup': 'Emirates NBA Cup', 'In-Season Tournament': 'Emirates NBA Cup', 'NBA Paris Game': 'NBA Paris Games',
       'SoFi Play-In Tournament': 'Play-In', 'East Play-In': 'Play-In', 'West Play-In': 'Play-In'}
mud = t.gameLabel.isin(PAD.keys())
t['gameLabel'] = t.gameLabel.replace(PAD)
log.append(('Rótulos de fase padronizados (gameLabel)', f'{int(mud.sum())}', 'ex.: "East First Round" → "East - First Round"; NBA Cup com um nome só'))

# 4. placar: soma de cestas (2×FG + 3P + LL) contra placar, quartos e soma dos jogadores
calc = 2 * t.fieldGoalsMade + t.threePointersMade + t.freeThrowsMade
q = t[['q1Points', 'q2Points', 'q3Points', 'q4Points']].sum(axis=1, min_count=4) + t.otAllPoints.fillna(0)
pj = j.groupby(['gameId', 'playerteamName']).points.sum()
pj = pd.Series([pj.get((g, n), np.nan) for g, n in zip(t.gameId, t.teamName)], index=t.index)
erro = (calc != t.teamScore) & ((calc == q) | (calc == pj))
for i in t.index[erro]:
    corr.append((t.at[i, 'gameId'], t.at[i, 'teamName'], 'teamScore', int(t.at[i, 'teamScore']), int(calc[i]),
                 'placar diferente da soma das cestas; quartos e/ou jogadores confirmam a soma'))
t.loc[erro, 'teamScore'] = calc[erro]
log.append(('Placares corrigidos pela soma das cestas', f'{int(erro.sum())} linhas', 'só quando quartos ou jogadores confirmam; o placar do adversário acompanha'))
# placar do adversário acompanha
op = t.set_index(['gameId', 'teamId']).teamScore
novo = pd.Series([op[(g, o)] for g, o in zip(t.gameId, t.opponentTeamId)], index=t.index)
t['opponentScore'] = novo

# 5. vitória pelo placar corrigido
w = (t.teamScore > t.opponentScore).astype(int)
dif = w != t.win
for i in t.index[dif]:
    corr.append((t.at[i, 'gameId'], t.at[i, 'teamName'], 'win', int(t.at[i, 'win']), int(w[i]), 'vitória refeita pelo placar corrigido'))
t['win'] = w
log.append(('Vitória refeita pelo placar corrigido', f'{int(dif.sum())} linhas', 'ex.: Raptors 102 x 105 Trail Blazers (13/11/2004) vinha como vitória do Raptors'))

# 6. rebotes: total = ofensivos + defensivos
rs = t.reboundsOffensive + t.reboundsDefensive
dr = rs != t.reboundsTotal
for i in t.index[dr]:
    corr.append((t.at[i, 'gameId'], t.at[i, 'teamName'], 'reboundsTotal', int(t.at[i, 'reboundsTotal']), int(rs[i]), 'total ≠ ofensivos + defensivos; jogadores confirmam'))
t.loc[dr, 'reboundsTotal'] = rs[dr]
log.append(('Rebotes totais = ofensivos + defensivos', f'{int(dr.sum())} linhas', 'confere com a soma dos jogadores'))

# 7. percentuais recalculados
for c in ['fieldGoals', 'threePointers', 'freeThrows']:
    a = t[c + 'Attempted']; t[c + 'Percentage'] = np.where(a > 0, t[c + 'Made'] / a.where(a > 0), np.nan)
log.append(('Percentuais de arremesso recalculados (XX,XX%)', 'todas', 'cestas ÷ tentativas; vazio quando não houve tentativa'))

# 8. temporada, data, Vitória/Mandante, observações
y = s.str[1:3].astype(int); y0 = np.where(y >= 90, 1900 + y, 2000 + y)
t.insert(t.columns.get_loc('gameDateTimeEst') + 1, 'season', [f'{a}-{str(a + 1)[-2:]}' for a in y0])
t['gameDateTimeEst'] = pd.to_datetime(t.gameDateTimeEst)
t.insert(t.columns.get_loc('gameDateTimeEst') + 1, 'gameDate', t.gameDateTimeEst.dt.normalize())
t['win'] = t.win.map({1: 'Vitória', 0: 'Derrota'}); t['home'] = t.home.map({1: 'Mandante', 0: 'Visitante'})
log.append(('Vitória/Derrota e Mandante/Visitante', 'todas', 'antes 1/0'))
log.append(('Temporada (season) pelo ID do jogo e data do jogo (gameDate)', 'todas', 'mesma regra da tabela de jogadores'))
bolha = set(j.loc[j.observacoes.str.contains('bolha', na=False), 'gameId'])
obs = pd.Series('', index=t.index)
obs[t.gameId.isin(bolha)] = 'Jogo na bolha da NBA (Orlando, 2020): sem público, quadra neutra'
sem_j = ~t.gameId.isin(set(j.gameId))
obs[sem_j] = 'Sem box score de jogadores na fonte; placar refeito pela soma das cestas'
t['observacoes'] = obs
log.append(('Jogos da bolha marcados em observacoes', f'{int(t.gameId.isin(bolha).sum())} linhas / {len(bolha)} jogos', 'mesmos jogos marcados nos jogadores'))
log.append(('Jogos sem jogadores na fonte marcados', f'{int(sem_j.sum())} linhas / {t.gameId[sem_j].nunique()} jogos', 'Jazz 128 x 112 Wizards (19/03/2025) e Magic 116 x 105 Spurs (01/04/2025); placar vinha 0 x 0'))

t = t.sort_values(['gameDateTimeEst', 'gameId', 'home'], ascending=[False, False, True]).reset_index(drop=True)
pd.to_pickle(t, 'dados/times_final.pkl')
pd.to_pickle((log, pd.DataFrame(corr, columns=['gameId', 'time', 'coluna', 'antes', 'depois', 'motivo'])), 'dados/times_log.pkl')
for l in log: print(l)
print(t.shape, t.gameId.nunique())
