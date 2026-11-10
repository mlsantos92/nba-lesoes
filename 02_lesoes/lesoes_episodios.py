"""Afastamentos (saída -> retorno da lista de lesionados) e jogos perdidos em cada um."""
import pandas as pd, numpy as np, re

P = pd.read_pickle('lesoes/pst_ligado.pkl')
j = pd.read_pickle('dados/jogadores_20t.pkl')
t = pd.read_pickle('dados/times_20t.pkl')

PARTES = [  # (palavras na fonte, parte do corpo)
    (r'concussion|head', 'Cabeça / concussão'), (r'achilles', 'Tendão de Aquiles'), (r'ankle', 'Tornozelo'), (r'knee|patella|acl|mcl|meniscus', 'Joelho'),
    (r'hamstring', 'Posterior da coxa'), (r'quad', 'Quadríceps'), (r'calf', 'Panturrilha'), (r'groin|adductor', 'Virilha'), (r'hip', 'Quadril'),
    (r'back|lumbar|spine', 'Costas'), (r'toe', 'Dedo do pé'), (r'foot|plantar|heel|metatarsal', 'Pé'), (r'shin|tibia|fibula|leg', 'Perna'),
    (r'thigh', 'Coxa'), (r'shoulder|rotator', 'Ombro'), (r'elbow', 'Cotovelo'), (r'wrist', 'Punho'), (r'thumb|finger|hand', 'Mão / dedo'),
    (r'neck', 'Pescoço'), (r'rib', 'Costela'), (r'abdom|oblique|core', 'Abdômen'), (r'eye|orbital', 'Olho'), (r'nose|face|jaw|facial|tooth|mouth', 'Rosto'),
    (r'chest|pectoral', 'Peito'), (r'arm|bicep|tricep', 'Braço'), (r'pelvi', 'Pelve'), (r'heart|cardiac', 'Coração')]

def classifica(nota):
    n = str(nota).lower()
    m = re.sub(r'^\s*placed on (il|ir|inactive list)( with| for| due to)?\s*', '', n).strip()
    m = re.sub(r'\((out for season|out indefinitely|dtd|date approximate)\)', '', m).strip()
    if m in ('', 'placed on il'): return 'Sem motivo informado', '', ''
    if re.search(r'health and safety|covid|protocol', m): return 'Protocolo de saúde (COVID)', '', ''
    if re.search(r'\brest\b|load management|maintenance', m): return 'Descanso', '', ''
    if re.search(r'personal|family|birth|bereavement|funeral|death', m): return 'Motivo pessoal', '', ''
    if re.search(r'suspen', m): return 'Suspensão', '', ''
    lado = 'Esquerdo' if 'left' in m else ('Direito' if 'right' in m else '')
    for pat, parte in PARTES:
        if re.search(pat, m): return 'Lesão', parte, lado
    if re.search(r'illness|flu|virus|sick|gastro|migraine|infection|strep|pneumonia|fever|dehydrat|food poison|mono|tonsil|appendic|cold\b', m):
        return 'Doença', '', ''
    if re.search(r'injur|surgery|sprain|strain|fracture|torn|tear|sore|bruise|contusion|tendin|dislocat|spasm', m): return 'Lesão', 'Outra / não especificada', lado
    return 'Outro', '', ''

P = P.dropna(subset=['personId']).sort_values(['personId', 'Date', 'evento'], ascending=[True, True, False])
fim_temp = t.groupby(['season', 'teamName']).gameDate.max()
ep, abertos, sem_saida = [], {}, 0
def temporada(d):
    if pd.Timestamp('2020-08-01') <= d <= pd.Timestamp('2020-11-30'): return '2019-20'  # bolha: 2019-20 terminou em outubro de 2020
    y = d.year if d.month >= 8 else d.year - 1
    return f'{y}-{str(y + 1)[-2:]}'
# eventos -> afastamentos: uma saída abre; retorno fecha; nova saída com afastamento aberto fecha o anterior e abre outro
for r in P.itertuples():
    pid = r.personId
    if r.evento == 'saida':
        if pid in abertos: ep.append((abertos.pop(pid), r, 'nova saída'))
        abertos[pid] = r
    else:
        if pid not in abertos: sem_saida += 1; continue
        ep.append((abertos.pop(pid), r, 'retorno'))
for pid, s0 in abertos.items(): ep.append((s0, None, None))

# primeiro jogo em quadra depois do início (qualquer time) e o time desse jogo
jd = j[['personId', 'gameDate', 'playerteamName']].drop_duplicates(['personId', 'gameDate']).sort_values('gameDate')
por_j = {p: (g.gameDate.values, g.playerteamName.values) for p, g in jd.groupby('personId')}
fim_temp = t.groupby(['season', 'teamName']).gameDate.max()
E = []
for s0, fecha, como in ep:
    tp = temporada(s0.Date)
    if tp < '2006-07': continue
    ini = s0.Date.normalize()
    d, tm = por_j.get(s0.personId, (np.array([], dtype='datetime64[ns]'), np.array([])))
    k = np.searchsorted(d, np.datetime64(ini))
    prim, time_prim = (pd.Timestamp(d[k]), tm[k]) if k < len(d) else (pd.NaT, None)
    reg = fecha.Date.normalize() if fecha is not None else pd.NaT
    if pd.notna(prim) and (pd.isna(reg) or prim < reg):
        fim, inferido = prim, 'Sim (volta em quadra antes do retorno registrado)'
    elif pd.notna(reg):
        fim, inferido = reg, ''
    else:
        lim = fim_temp.get((tp, s0.Team), pd.NaT); fim, inferido = (lim + pd.Timedelta(days=1)), 'Sem retorno: considerado até o fim da temporada'
    time_depois = fecha.Team if (fecha is not None and como == 'retorno') else (time_prim or s0.Team)
    tipo, parte, lado = classifica(s0.Notes)
    E.append(dict(personId=s0.personId, jogador=s0.playerKey, time=s0.Team, timeTemporadasSeguintes=time_depois, temporada=tp, inicio=ini,
                  retornoRegistrado=reg if (fecha is not None and como == 'retorno') else pd.NaT, fimConsiderado=fim,
                  temporadaRetorno=temporada(fim) if pd.notna(fim) else '', retornoInferido=inferido,
                  motivoFonte=str(s0.Notes).strip(), tipo=tipo, parteDoCorpo=parte, lado=lado,
                  foraDaTemporada='Sim' if 'out for season' in str(s0.Notes).lower() else '', fonte=s0.fonte))
E = pd.DataFrame(E).sort_values(['inicio', 'time', 'jogador']).reset_index(drop=True)
E['trocaDuranteAfastamento'] = np.where(E.time != E.timeTemporadasSeguintes, 'Possível (voltou por outro time)', '')
E.insert(0, 'afastamentoId', np.arange(1, len(E) + 1))
print('retornos inferidos pelo box score:', int(E.retornoInferido.str.startswith('Sim').sum()), '| sem retorno:', int(E.retornoInferido.str.startswith('Sem').sum()))

# jogos perdidos: na temporada do início, jogos do time do afastamento; nas seguintes, do time em que ele voltou
tg = t[['gameId', 'gameDate', 'season', 'teamName', 'opponentTeamName', 'home', 'gameType']].rename(columns={'teamName': 'time'})
cols = ['afastamentoId', 'personId', 'jogador', 'time', 'timeTemporadasSeguintes', 'temporada', 'inicio', 'fimConsiderado', 'tipo', 'parteDoCorpo', 'lado', 'motivoFonte']
M1 = E[cols].merge(tg, left_on=['time', 'temporada'], right_on=['time', 'season'])
Ex = E[E.temporadaRetorno > E.temporada]
print('afastamentos que atravessam temporadas:', len(Ex))
M2 = Ex[cols].drop(columns='time').rename(columns={'timeTemporadasSeguintes': 'time'}).merge(tg, on='time')
M2 = M2[M2.season > M2.temporada]
# temporada seguinte só conta se o jogador estava na liga na anterior (ou ela é a do início) e voltou a jogar na seguinte
temps = sorted(t.season.unique()); idx = {x: i for i, x in enumerate(temps)}
jogou = set(zip(j.personId, j.season))
ant = lambda x: temps[idx[x] - 1] if idx[x] > 0 else None
prox = lambda x: temps[idx[x] + 1] if idx[x] + 1 < len(temps) else None
ok = [((ant(se) == ti) or ((p, ant(se)) in jogou)) and (prox(se) is None or (p, prox(se)) in jogou or (p, se) in jogou)
      for p, se, ti in zip(M2.personId, M2.season, M2.temporada)]
M2 = M2[ok]
M = pd.concat([M1.drop(columns='timeTemporadasSeguintes'), M2], ignore_index=True)
M = M[(M.gameDate >= M.inicio) & (M.gameDate < M.fimConsiderado)].rename(columns={'season': 'temporadaJogo'}).drop(columns='timeTemporadasSeguintes', errors='ignore')
# validação: o jogador aparece jogando em algum desses jogos?
jog = set(zip(j.gameId, j.personId))
M['jogouMesmoAssim'] = [(g, p) in jog for g, p in zip(M.gameId, M.personId)]
E = E.merge(M.groupby('afastamentoId').size().rename('jogosPerdidos'), on='afastamentoId', how='left').fillna({'jogosPerdidos': 0})
E['jogosPerdidos'] = E.jogosPerdidos.astype(int)
print('afastamentos', len(E), '| retornos sem saída registrada', sem_saida)
print(E.tipo.value_counts().to_string()); print(E[E.tipo == 'Lesão'].parteDoCorpo.value_counts().head(12).to_string())
print('jogos perdidos', len(M), '| jogou mesmo assim', int(M.jogouMesmoAssim.sum()), f'({M.jogouMesmoAssim.mean():.2%})')
print(M.groupby('temporadaJogo').size().to_dict())
import sys; sys.path.insert(0, '/tmp/claude-0/-home-claude/898d6bc3-1107-5693-9ccd-56f3d98ed7ab/scratchpad'); import categorias as C
cat = [C.categoria(n, t) for n, t in zip(E.motivoFonte, E.tipo)]
E['categoria'] = [a for a, b in cat]; E['detalhe'] = [b for a, b in cat]
M = M.merge(E[['afastamentoId', 'categoria', 'detalhe']], on='afastamentoId')
pd.to_pickle((E, M), 'lesoes/episodios.pkl')
