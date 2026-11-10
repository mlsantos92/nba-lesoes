"""Liga as estatísticas de arremesso e avançadas do Basketball-Reference (por jogador-temporada) à base."""
import pandas as pd, numpy as np, re, unicodedata, sys

raw = pd.read_csv(sys.argv[1])
j = pd.read_pickle('dados/jogadores_20t_lesoes.pkl')
def norm(s):
    s = unicodedata.normalize('NFKD', str(s).replace('ı', 'i')).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"[.'`*]", '', s).replace('-', ' ')
    return ' '.join(x for x in s.split() if x not in ('jr', 'sr', 'ii', 'iii', 'iv'))
ALIAS = {'kj martin': 'kenyon martin', 'ron holland': 'ronald holland', 'flip murray': 'ronald murray', 'jeff taylor': 'jeffery taylor', 'yi jianlian': 'jianlian yi', 'nene': 'nene hilario', 'ishmael smith': 'ish smith', 'louis williams': 'lou williams', 'jose juan barea': 'jj barea'}
raw['season'] = raw.ano.map(lambda y: f'{y - 1}-{str(y)[-2:]}')
raw['n'] = raw.jogador.map(norm).replace(ALIAS)
# uma linha por jogador-temporada: a de total quando foi trocado
tot = raw.team.isin(['TOT', '2TM', '3TM', '4TM', '5TM'])
raw['prio'] = np.where(tot, 0, 1)
raw = raw.sort_values(['season', 'n', 'prio', 'mp'], ascending=[True, True, True, False]).drop_duplicates(['season', 'n', 'age'])
r = j[j.gameType == 'Regular Season']
cat = r.groupby(['personId', 'season']).agg(nome=('playerKey', 'first'), fn=('firstName', 'first'), ln=('lastName', 'first'), jogos=('gameId', 'nunique')).reset_index()
cat['n'] = (cat.fn.fillna('') + ' ' + cat.ln.fillna('')).map(norm).replace(ALIAS)
m = cat.merge(raw, on=['season', 'n'], how='left', suffixes=('', '_bb'))
# homônimos na mesma temporada: fica o par com número de jogos mais próximo
m['dif'] = (m.jogos - pd.to_numeric(m.games, errors='coerce')).abs()
m = m.sort_values('dif').drop_duplicates(['personId', 'season']).drop_duplicates(['season', 'n', 'age'], keep='first')
print('jogador-temporadas na base:', len(cat), '| ligadas ao bbref:', int(m.avg_dist.notna().sum() + m.ast_pct.notna().sum() > 0), int(m.games.notna().sum()))
print('diferença de jogos (bbref x base):', m.dif.describe().round(1).to_dict())
sem = m[m.games.isna()].sort_values('jogos', ascending=False)[['season', 'nome', 'jogos']].head(15); print(sem.to_string())
num = ['avg_dist', 'pct_fga_00_03', 'pct_fga_03_10', 'pct_fga_10_16', 'pct_fga_16_xx', 'pct_fga_fg3a', 'fg_pct_00_03', 'pct_ast_fg2', 'pct_fga_dunk',
       'ast_pct', 'stl_pct', 'blk_pct', 'drb_pct', 'orb_pct', 'dbpm', 'dws', 'usg_pct', 'tov_pct', 'games', 'mp']
for c in num: m[c] = pd.to_numeric(m[c], errors='coerce')
m[['personId', 'season', 'team', 'pos'] + num].to_pickle('lesoes/bbref_estilo.pkl')
