"""Liga os registros de lesão (Pro Sports Transactions) aos jogadores da base pelo nome, time e temporada."""
import pandas as pd, numpy as np, re, unicodedata

P = pd.read_pickle('lesoes/pst_2006_2026.pkl').copy()
j = pd.read_pickle('dados/jogadores_20t.pkl')
P['Team'] = P.Team.replace({'Blazers': 'Trail Blazers', 'Sonics': 'SuperSonics'})

def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"[.'`]", '', s).replace('-', ' ')
    w = [x for x in s.split() if x not in ('jr', 'sr', 'ii', 'iii', 'iv')]
    return ' '.join(w)

def variantes(campo):
    """'(William) Tony Parker' -> {tony parker, william tony parker}; 'A / B' -> ambos."""
    out = set()
    for parte in str(campo).split(' / '):
        parte = parte.strip()
        if not parte or parte == 'nan': continue
        out.add(norm(re.sub(r'\([^)]*\)', '', parte)))
        out.add(norm(parte.replace('(', '').replace(')', '')))
    ALIAS = {'dj augustine': 'dj augustin', 'stanislav medvedenko': 'slava medvedenko', 'sun yue': 'yue sun', 'roy devyn marble': 'devyn marble',
             'roy marble devyn': 'devyn marble', 'roy marble': 'devyn marble'}
    out |= {ALIAS[x] for x in out if x in ALIAS}
    return {x for x in out if x}

def temporada(d):
    if pd.Timestamp('2020-08-01') <= d <= pd.Timestamp('2020-11-30'): return '2019-20'  # bolha: 2019-20 terminou em outubro de 2020
    y = d.year if d.month >= 8 else d.year - 1
    return f'{y}-{str(y + 1)[-2:]}'

# catálogo: nome normalizado -> (personId, playerKey) por time e temporada
j['_n'] = (j.firstName.fillna('') + ' ' + j.lastName.fillna('')).map(norm)
cat = j.groupby(['_n', 'personId', 'playerKey', 'playerteamName', 'season']).size().reset_index()[['_n', 'personId', 'playerKey', 'playerteamName', 'season']]
por_nome = {}
for r in cat.itertuples():
    por_nome.setdefault(r._1, []).append(r)
temps = sorted(j.season.unique()); ti = {t: i for i, t in enumerate(temps)}

P['jogadorFonte'] = P.Relinquished.where(P.Relinquished.notna(), P.Acquired)
P['evento'] = np.where(P.Relinquished.notna(), 'saida', 'retorno')
P['temporada'] = P.Date.map(temporada)
res = []
for r in P.itertuples():
    cand = []
    for v in variantes(r.jogadorFonte):
        cand += por_nome.get(v, [])
    s = ti.get(r.temporada)
    def filtra(cond):
        return {(c.personId, c.playerKey) for c in cand if cond(c)}
    m = filtra(lambda c: c.playerteamName == r.Team and c.season == r.temporada)
    nivel = 'time+temporada'
    if len(m) != 1:
        m = filtra(lambda c: c.playerteamName == r.Team and s is not None and abs(ti[c.season] - s) <= 1); nivel = 'time+temporada vizinha'
    if len(m) != 1:
        m = filtra(lambda c: s is not None and abs(ti[c.season] - s) <= 1); nivel = 'temporada (outro time)'
    if len(m) == 1:
        pid, key = next(iter(m)); res.append((pid, key, nivel))
    else:
        res.append((np.nan, None, 'sem ligação' if not m else 'ambíguo'))
P[['personId', 'playerKey', 'ligacaoNome']] = pd.DataFrame(res, index=P.index)
print(P.ligacaoNome.value_counts().to_string())
sem = P[P.personId.isna()]
print(sem.jogadorFonte.value_counts().head(25).to_string())
pd.to_pickle(P, 'lesoes/pst_ligado.pkl')
