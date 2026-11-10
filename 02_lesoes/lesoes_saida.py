"""Monta as saídas: planilha de lesões, e colunas de afastamento nas tabelas de jogadores e times (20 temporadas)."""
import pandas as pd, numpy as np

E, M = pd.read_pickle('lesoes/episodios.pkl')
P = pd.read_pickle('lesoes/pst_ligado.pkl')
j = pd.read_pickle('dados/jogadores_20t.pkl').copy()
t = pd.read_pickle('dados/times_20t.pkl').copy()

def rotulo(r):
    if r.tipo == 'Lesão':
        return f"{r.categoria} - {r.parteDoCorpo}{(' ' + r.lado.lower()) if r.lado else ''}"
    return r.categoria

# jogadores: marca o jogo de retorno de cada afastamento com jogos perdidos
jd = j[['personId', 'gameDate']].drop_duplicates().sort_values('gameDate')
por = {p: g.gameDate.values for p, g in jd.groupby('personId')}
ret = []
for r in E[E.jogosPerdidos > 0].itertuples():
    d = por.get(r.personId)
    if d is None: continue
    k = np.searchsorted(d, np.datetime64(r.fimConsiderado))
    if k < len(d) and (pd.Timestamp(d[k]) - r.fimConsiderado).days <= 400:
        ret.append((r.personId, pd.Timestamp(d[k]), r.afastamentoId, r.jogosPerdidos, rotulo(r)))
R = pd.DataFrame(ret, columns=['personId', 'gameDate', 'afastamentoId', 'jogosPerdidosAntes', 'motivoAfastamento']).drop_duplicates(['personId', 'gameDate'], keep='last')
j = j.drop(columns=[c for c in ['retornoDeAfastamento', 'afastamentoId', 'jogosPerdidosAntes', 'motivoAfastamento'] if c in j.columns])
j = j.merge(R, on=['personId', 'gameDate'], how='left')
j.insert(j.columns.get_loc('afastamentoId'), 'retornoDeAfastamento', np.where(j.afastamentoId.notna(), 'Sim', ''))
ctrl = ['corrigidoX10', 'ligacao', 'blocoJogo', 'alertaMinutos', 'observacoes']
novos = ['retornoDeAfastamento', 'afastamentoId', 'jogosPerdidosAntes', 'motivoAfastamento']
j = j[[c for c in j.columns if c not in novos + ctrl] + novos + ctrl]

# times: afastados em cada jogo
M['rotulo'] = [rotulo(r) for r in M.itertuples()]
A = M.groupby(['gameId', 'time']).apply(lambda g: pd.Series({'afastados': len(g), 'afastadosLista': '; '.join(f"{a.split(' (')[0]} ({b})" for a, b in zip(g.jogador, g.rotulo))}),
                                         include_groups=False).reset_index()
t = t.drop(columns=[c for c in ['afastados', 'afastadosLista'] if c in t.columns]).merge(A, left_on=['gameId', 'teamName'], right_on=['gameId', 'time'], how='left').drop(columns='time')
t['afastados'] = t.afastados.fillna(0).astype(int); t['afastadosLista'] = t.afastadosLista.fillna('')
t = t[[c for c in t.columns if c not in ('afastados', 'afastadosLista', 'observacoes')] + ['afastados', 'afastadosLista', 'observacoes']]

pd.to_pickle(j, 'dados/jogadores_20t_lesoes.pkl'); pd.to_pickle(t, 'dados/times_20t_lesoes.pkl')
print('jogos de retorno marcados:', int((j.retornoDeAfastamento == 'Sim').sum()), '| time-jogos com afastados:', int((t.afastados > 0).sum()), 'de', len(t))
print(t.afastados.describe().round(2).to_dict())
