"""Aplica o box score oficial (Basketball-Reference) nos jogos com divergência time x jogadores."""
import pandas as pd, numpy as np, json, unicodedata, sys

res = json.load(open('sumulas/bbref_res.json'))
L = {x['g']: x for x in json.load(open('sumulas/bbref_lista.json'))}
j = pd.read_pickle('dados/jogadores_final_v3.pkl').copy()
t = pd.read_pickle('dados/times_final_v2.pkl').copy()
norm = lambda s: unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower().replace('.', '').replace("'", '').replace('-', ' ').strip()
# box: fg fga fg3 fg3a ft fta orb drb trb ast stl blk tov pf pts
TCOL = ['fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted',
        'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'teamScore']
PCOL = TCOL[:-1] + ['points']
num = lambda v: float(v) if v not in ('', None) else np.nan
mud = []  # tabela, gameId, time, jogador, coluna, antes, depois
falta = []
for gs, box in res.items():
    g = int(gs); info = L[g]
    for code, nome in [(info['h'], info['hn']), (info['a'], info['an'])]:
        if code not in box: falta.append((g, nome, 'time não achado no box')); continue
        b = box[code]; tot = [num(v) for v in b['tot'].split('|')]
        # time
        i = t.index[(t.gameId == g) & (t.teamName == nome)][0]
        for c, v in zip(TCOL, tot):
            if c == 'turnovers' or np.isnan(v): continue
            if t.at[i, c] != v:
                mud.append(('times', g, nome, '', c, t.at[i, c], v)); t.at[i, c] = v
        # jogadores
        of = {}
        for r in b['rows']:
            f = r.split('|'); of[norm(f[0])] = (f[0], f[1], [num(x) for x in f[2:]])
        nos = j.index[(j.gameId == g) & (j.playerteamName == nome)]
        vistos = set()
        for k in nos:
            nm = norm(f"{j.at[k, 'firstName']} {j.at[k, 'lastName']}")
            alvo = nm if nm in of else next((o for o in of if o.split()[-1] == nm.split()[-1] and o[0] == nm[0]), None)
            if alvo is None:
                mud.append(('jogadores', g, nome, f"{j.at[k, 'firstName']} {j.at[k, 'lastName']}", 'linha removida', j.at[k, 'points'], 'não está no box oficial'))
                j = j.drop(k); continue
            vistos.add(alvo); _, mp, vals = of[alvo]
            for c, v in zip(PCOL, vals):
                if np.isnan(v): continue
                if j.at[k, c] != v:
                    mud.append(('jogadores', g, nome, of[alvo][0], c, j.at[k, c], v)); j.at[k, c] = v
        for o in set(of) - vistos:
            falta.append((g, nome, f'jogador do box oficial sem linha na base: {of[o][0]} ({of[o][2][-1]:.0f} pts)'))
m = pd.DataFrame(mud, columns=['tabela', 'gameId', 'time', 'jogador', 'coluna', 'antes', 'depois'])
# recalcula percentuais, total de rebotes e placar do adversário
for df, pts in [(j, 'points'), (t, 'teamScore')]:
    for c in ['fieldGoals', 'threePointers', 'freeThrows']:
        a = df[c + 'Attempted']; df[c + 'Percentage'] = np.where(a > 0, df[c + 'Made'] / a.where(a > 0), np.nan)
op = t.set_index(['gameId', 'teamId']).teamScore
t['opponentScore'] = [op[(g, o)] for g, o in zip(t.gameId, t.opponentTeamId)]
print(m.groupby(['tabela', 'coluna']).size().to_string()); print('pendentes:', falta)
pd.to_pickle(j.reset_index(drop=True), 'dados/jogadores_final_v4.pkl'); pd.to_pickle(t, 'dados/times_final_v3.pkl'); pd.to_pickle((m, falta), 'dados/oficial_mud.pkl')
