"""Preenche as avançadas dos jogos com asterisco com as fórmulas que reproduzem a fonte (>= 90% dos jogos de 2024-25)."""
import sys; sys.path.insert(0, '/tmp/claude-0/-home-claude/898d6bc3-1107-5693-9ccd-56f3d98ed7ab/scratchpad')
import pandas as pd, numpy as np, avancadas as A

t = pd.read_pickle('dados/times_final_v4.pkl').copy()
j = pd.read_pickle('dados/jogadores_final_v5.pkl').copy()
JT = [22400524, 22400532, 22400537, 22400538, 22400617]          # times: jogos incluídos
JJ = [22400532, 22400537, 22400538, 22400617, 22400627, 22400988]  # jogadores: jogos montados pela súmula

# candidatos extras: colunas "estimadas" (posses = FGA - OREB + TOV + 0,44 FTA)
o = t.set_index(['gameId', 'teamId']); O = o.loc[list(zip(t.gameId, t.opponentTeamId))].reset_index(drop=True); O.index = t.index
def poss(X, tov): return X.fieldGoalsAttempted - X.reboundsOffensive + tov + 0.44 * X.freeThrowsAttempted
rt = A.times(t)
for nome, tov_p, tov_o in [('', t.turnovers, O.turnovers), ('_tot', t.turnovers + t.turnoversTeam.fillna(0), O.turnovers + O.turnoversTeam.fillna(0))]:
    p1, p2 = poss(t, tov_p), poss(O, tov_o)
    rt['estimatedOffensiveRating' + nome] = 100 * A.div(t.teamScore, p1)
    rt['estimatedDefensiveRating' + nome] = 100 * A.div(O.teamScore, p1)
    rt['estimatedDefensiveRating2' + nome] = 100 * A.div(O.teamScore, p2)
    rt['estimatedNetRating' + nome] = rt['estimatedOffensiveRating' + nome] - rt['estimatedDefensiveRating2' + nome]
    rt['estimatedPace' + nome] = 48 * A.div((p1 + p2) / 2, t.numMinutes / 5)
    rt['assistToTurnoverRatio' + nome] = A.div(t.assists, tov_p)
    rt['teamTurnoverPercentage' + nome] = A.div(tov_p, p1)
rj = A.jogadores(j[j.season == '2024-25'], t)
RND = lambda c: 1 if ('Rating' in c or 'Pace' in c or c in ('pace', 'pacePer40', 'possessions', 'assistRatio')) else 3

def valida(src, calc, c):
    m = src.notna() & pd.Series(calc, index=src.index).notna()
    if m.sum() < 200: return 0.0, int(m.sum())
    d = (src[m].astype(float) - np.round(pd.Series(calc, index=src.index)[m], RND(c))).abs()
    return float((d <= (0.15 if RND(c) == 1 else 0.0015)).mean()), int(m.sum())

rel = []
# times
v = t.season == '2024-25'
alvo_t = t.gameId.isin(JT)
for c in rt.columns:
    base = c.replace('_tot', '').replace('2', '') if c.startswith('estimatedDefensiveRating2') else c.replace('_tot', '')
    if base not in t.columns: continue
    taxa, n = valida(t.loc[v, base], rt.loc[v, c], base)
    rel.append(('times', base, c, round(taxa, 3), n))
R = pd.DataFrame(rel, columns=['tabela', 'coluna', 'formula', 'acerto', 'n'])
melhor_t = R.sort_values('acerto', ascending=False).drop_duplicates('coluna')
# jogadores
vj = j.season == '2024-25'
rel = []
for c in rj.columns:
    taxa, n = valida(j.loc[vj, c], rj[c], c)
    rel.append(('jogadores', c, c, round(taxa, 3), n))
melhor_j = pd.DataFrame(rel, columns=['tabela', 'coluna', 'formula', 'acerto', 'n'])
tudo = pd.concat([melhor_t, melhor_j]).sort_values(['tabela', 'acerto'], ascending=[True, False])
print(tudo.to_string(index=False))

usa_t = melhor_t[melhor_t.acerto >= 0.9]; usa_j = melhor_j[melhor_j.acerto >= 0.9]
mud = []
for r in usa_t.itertuples():
    vals = np.round(rt.loc[alvo_t, r.formula], RND(r.coluna))
    antes = t.loc[alvo_t, r.coluna].isna()
    t.loc[alvo_t & t[r.coluna].isna(), r.coluna] = vals[antes.values]
    mud.append(('times', r.coluna, int(antes.sum()), r.acerto))
alvo_j = j.gameId.isin(JJ)
rja = A.jogadores(j[alvo_j], t)
for r in usa_j.itertuples():
    vals = np.round(rja[r.coluna], RND(r.coluna))
    vazio = j.loc[alvo_j, r.coluna].isna()
    j.loc[vazio[vazio].index, r.coluna] = vals[vazio.values]
    mud.append(('jogadores', r.coluna, int(vazio.sum()), r.acerto))
M = pd.DataFrame(mud, columns=['tabela', 'coluna', 'preenchidas', 'acerto_validacao'])
print(M.to_string(index=False))
nota_t = '; avançadas calculadas pelo box (fórmulas que reproduzem a fonte em 90% ou mais dos jogos de 2024-25); ratings e ritmo do lance a lance seguem vazios'
t.loc[alvo_t, 'observacoes'] = t.loc[alvo_t, 'observacoes'].str.replace('; colunas avançadas vazias', nota_t, regex=False)
j.loc[alvo_j, 'observacoes'] = ('* Jogo montado pela súmula oficial da NBA: avançadas de arremesso e distribuição de pontos calculadas pelo box; '
                                'sem ratings, uso, percentuais de equipe, tocos e faltas sofridos, posses e pontos por tipo de jogada')
pd.to_pickle(t, 'dados/times_final_v5.pkl'); pd.to_pickle(j, 'dados/jogadores_final_v6.pkl'); pd.to_pickle((tudo, M), 'dados/preenche_asterisco.pkl')
print(t.loc[alvo_t, 'observacoes'].iloc[0])
