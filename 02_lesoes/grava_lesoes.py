"""Planilha de lesões e afastamentos 2006-07 a 2025-26."""
import pandas as pd, numpy as np, xlsxwriter, math

E, M = pd.read_pickle('lesoes/episodios.pkl')
P = pd.read_pickle('lesoes/pst_ligado.pkl')
VAL = [l.split('|') for l in open('lesoes/validacao.txt').read().strip().split('\n')]

# jogos perdidos legíveis; atenção para possível troca na temporada do início
M = M.merge(E[['afastamentoId', 'trocaDuranteAfastamento']], on='afastamentoId')
M['atencao'] = np.where((M.trocaDuranteAfastamento != '') & (M.temporadaJogo == M.temporada), 'Voltou por outro time: conferir se já tinha sido trocado', '')
JP = M[['gameId', 'gameDate', 'temporadaJogo', 'gameType', 'time', 'opponentTeamName', 'home', 'jogador', 'personId', 'afastamentoId', 'tipo', 'categoria', 'detalhe', 'parteDoCorpo', 'lado', 'motivoFonte', 'atencao']]
JP = JP.rename(columns={'gameDate': 'data', 'temporadaJogo': 'temporada', 'gameType': 'tipoJogo', 'opponentTeamName': 'adversario', 'home': 'mando', 'tipo': 'tipoAfastamento'})
JP = JP.sort_values(['data', 'gameId', 'time', 'jogador'])
AF = E.rename(columns={'tipo': 'tipoAfastamento', 'time': 'timeNoInicio', 'timeTemporadasSeguintes': 'timeNoRetorno', 'fimConsiderado': 'volta'})
AF = AF[['afastamentoId', 'jogador', 'personId', 'temporada', 'timeNoInicio', 'inicio', 'volta', 'retornoRegistrado', 'retornoInferido', 'timeNoRetorno',
         'trocaDuranteAfastamento', 'jogosPerdidos', 'tipoAfastamento', 'categoria', 'detalhe', 'parteDoCorpo', 'lado', 'foraDaTemporada', 'motivoFonte', 'fonte']]

R1 = E.pivot_table(index='temporada', columns='tipo', values='afastamentoId', aggfunc='count', fill_value=0)
R1['Total de afastamentos'] = R1.sum(axis=1)
R1['Jogos perdidos (todos os motivos)'] = M.groupby('temporadaJogo').size()
R1['Jogos perdidos por lesão'] = M[M.tipo == 'Lesão'].groupby('temporadaJogo').size()
R1 = R1.fillna(0).astype(int).reset_index()
L = E[E.tipo == 'Lesão']
R2 = L.groupby('parteDoCorpo').agg(afastamentos=('afastamentoId', 'count'), jogosPerdidos=('jogosPerdidos', 'sum'),
                                   mediaJogosPorAfastamento=('jogosPerdidos', 'mean'), foraDaTemporada=('foraDaTemporada', lambda s: (s == 'Sim').sum()))
R2 = R2.sort_values('jogosPerdidos', ascending=False).reset_index(); R2['mediaJogosPorAfastamento'] = R2.mediaJogosPorAfastamento.round(1)
R3 = M[M.tipo == 'Lesão'].groupby(['jogador', 'personId']).agg(jogosPerdidos=('gameId', 'count'), afastamentos=('afastamentoId', 'nunique'),
                                                                 temporadas=('temporadaJogo', lambda s: f"{s.min()} a {s.max()}")).sort_values('jogosPerdidos', ascending=False).head(100).reset_index()
ORD = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Doença', 'Lesão sem diagnóstico', 'Protocolo de saúde (COVID)', 'Descanso', 'Motivo pessoal', 'Suspensão', 'Outro', 'Sem motivo informado']
RC = E.groupby('categoria').agg(afastamentos=('afastamentoId', 'count'), jogosPerdidos=('jogosPerdidos', 'sum'), mediaJogosPorAfastamento=('jogosPerdidos', 'mean'),
                                 foraDaTemporada=('foraDaTemporada', lambda s: (s == 'Sim').sum())).reindex(ORD).dropna(how='all').reset_index()
RC['mediaJogosPorAfastamento'] = RC.mediaJogosPorAfastamento.round(1)
RC['% dos jogos perdidos'] = (RC.jogosPerdidos / RC.jogosPerdidos.sum()).round(4)
RD = E[E.categoria.isin(ORD[:6])].groupby(['categoria', 'detalhe']).agg(afastamentos=('afastamentoId', 'count'), jogosPerdidos=('jogosPerdidos', 'sum'),
                                                                        mediaJogosPorAfastamento=('jogosPerdidos', 'mean')).reset_index()
RD['mediaJogosPorAfastamento'] = RD.mediaJogosPorAfastamento.round(1)
RD['o'] = RD.categoria.map({c: i for i, c in enumerate(ORD)}); RD = RD.sort_values(['o', 'jogosPerdidos'], ascending=[True, False]).drop(columns='o')
RT = M.pivot_table(index='temporadaJogo', columns='categoria', values='gameId', aggfunc='count', fill_value=0)
RT = RT[[c for c in ORD if c in RT.columns]].reset_index().rename(columns={'temporadaJogo': 'temporada'})
RP = E[E.categoria.isin(['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico'])].pivot_table(index='parteDoCorpo', columns='categoria', values='jogosPerdidos', aggfunc='sum', fill_value=0)
RP = RP[[c for c in ORD if c in RP.columns]]; RP['Total'] = RP.sum(axis=1); RP = RP.sort_values('Total', ascending=False).reset_index()
FON = P[['Date', 'Team', 'jogadorFonte', 'evento', 'Notes', 'personId', 'playerKey', 'ligacaoNome', 'fonte']].rename(columns={
    'Date': 'data', 'Team': 'time', 'Notes': 'nota', 'playerKey': 'jogadorNaBase'}).sort_values('data')
FON['evento'] = FON.evento.map({'saida': 'Saída (lista de lesionados)', 'retorno': 'Retorno'})

MET = [('Tema', 'Como foi feito'),
 ('Fonte', 'Pro Sports Transactions (prosportstransactions.com), entradas e saídas da lista de lesionados com o motivo. O site bloqueia acesso automatizado; usei três extrações públicas do mesmo site: Kaggle "NBA Injury Stats 1951-2023" (Logan Lauton, cópia no GitHub YohaanChokhany/cs441-nba-risk-contract-prediction) até out/2016, e GitHub joshcannonai/nba-game-intelligence-agent de out/2016 a mai/2026. No período em que se sobrepõem (2016-2023), 99% dos registros são idênticos.'),
 ('Ligação com a base', '99,7% dos registros ligados a um jogador da base pelo nome + time + temporada (com apelidos e grafias alternativas). Os 0,3% restantes são de jogadores que não entraram em quadra nessas 20 temporadas.'),
 ('Afastamento', 'Começa no registro de saída e termina no primeiro destes: retorno registrado, nova saída (outra lesão) ou primeiro jogo em que o jogador volta a entrar em quadra (retorno inferido pelo box score). Sem nenhum deles, vai até o fim da temporada.'),
 ('Jogos perdidos', 'Jogos do time entre o início e a volta. Na temporada do início, conta o time em que foi afastado; nas seguintes, o time pelo qual voltou (ex.: Kevin Durant em 2019-20 conta pelo Nets). Nenhum jogo perdido coincide com jogo em que o jogador entrou em quadra.'),
 ('Categorias', 'Pelo texto do motivo. Ligamento = entorse (sprain), rompimento de ligamento (ACL, MCL etc.), luxação, tendão de Aquiles, menisco, tendinite, fascite plantar, cartilagem e labrum (inclui desgaste). Muscular = estiramento (strain), contratura, espasmo, câimbra e lesões de posterior da coxa, panturrilha, virilha e quadríceps. Fratura = fratura, osso quebrado e fratura por estresse. Contusão (batidas) = pancadas (bruise/contusion), dores e inflamações (sore, pain, bursitis) e concussão. Doença = gripe, vírus, infecções etc. Lesão sem diagnóstico = a fonte só informa a região ("knee injury"). COVID fica separado de doença porque o protocolo também afastava quem só teve contato.'),
 ('Tipo e parte do corpo', 'Classificados a partir do texto do motivo (ex.: "sprained left ankle" -> Lesão, Tornozelo, Esquerdo). "Sem motivo informado" é comum antes de 2017, quando a lista de inativos também era usada por decisão técnica.'),
 ('Validação externa', '40 jogos perdidos sorteados e conferidos no box score do Basketball-Reference: 36 (90%) confirmados com o jogador inativo ou sem jogar. Os 4 restantes: 1 erro de temporada da bolha de 2020 (já corrigido), 1 troca durante a lesão, 1 suspensão e 1 provável dispensa.'),
 ('Limitações', '1) Trocas durante a lesão: a fonte não traz a data da troca; jogos do antigo time depois da troca podem ser contados (sinalizados na coluna "atencao"). 2) Suspensões e dispensas registradas como saída sem motivo podem aparecer como afastamento. 3) Registros ausentes na fonte não são recuperados (ex.: o rompimento do ligamento do joelho de Klay Thompson em jun/2019, por isso 2019-20 dele não aparece). 4) Antes de 2017 parte dos afastamentos "sem motivo" pode ser decisão técnica, não lesão.'),
 ('Ligação com as outras planilhas', 'Jogadores: colunas retornoDeAfastamento, afastamentoId, jogosPerdidosAntes e motivoAfastamento no jogo de volta. Times: colunas afastados e afastadosLista em cada jogo. A chave entre as três é gameId + personId e o afastamentoId.')]

wb = xlsxwriter.Workbook('/mnt/user-data/outputs/NBA_Lesoes_2006_2026.xlsx')
fh = wb.add_format({'bold': True}); fd = wb.add_format({'num_format': 'dd/mm/yyyy'}); fw = wb.add_format({'text_wrap': True, 'valign': 'top'})
def aba(nome, df, larg=None):
    ws = wb.add_worksheet(nome); ws.write_row(0, 0, list(df.columns), fh); ws.freeze_panes(1, 0)
    for i, row in enumerate(df.itertuples(index=False, name=None), 1):
        for k, x in enumerate(row):
            if x is None or x is pd.NaT or (isinstance(x, float) and math.isnan(x)): continue
            if isinstance(x, pd.Timestamp): ws.write_datetime(i, k, x.to_pydatetime(), fd); continue
            if isinstance(x, (np.integer,)): x = int(x)
            if isinstance(x, (np.floating, float)): x = float(x); x = int(x) if x.is_integer() else x
            ws.write(i, k, x)
    ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    for k, w in (larg or {}).items(): ws.set_column(k, k, w)
    return ws
aba('Afastamentos', AF, {1: 24, 5: 11, 6: 11, 7: 11, 8: 22, 12: 18, 13: 16, 16: 50, 17: 26})
aba('Jogos perdidos', JP, {1: 11, 7: 24, 13: 45, 14: 30})
aba('Por categoria', RC, {0: 26})
aba('Categoria e detalhe', RD, {0: 16, 1: 34})
aba('Jogos perdidos por categoria', RT)
aba('Categoria x parte do corpo', RP, {0: 24})
aba('Resumo por temporada', R1)
aba('Lesões por parte do corpo', R2, {0: 24})
aba('Mais jogos perdidos (lesão)', R3, {0: 26})
aba('Registros da fonte', FON, {0: 11, 2: 30, 4: 50, 8: 28})
v = wb.add_worksheet('Validação')
v.write_row(0, 0, ['gameId', 'jogador', 'Basketball-Reference'], fh)
for i, l in enumerate(VAL, 1): v.write_row(i, 0, [int(l[0]), l[1], l[2].replace('INATIVO', 'Inativo (confirmado)').replace('nao_encontrado', 'Não listado').replace('DNP: ', 'Não jogou: ')])
v.set_column(1, 2, 26)
m = wb.add_worksheet('Metodologia')
for i, l in enumerate(MET): m.write_row(i, 0, l, fh if i == 0 else fw)
m.set_column(0, 0, 24); m.set_column(1, 1, 120)
wb.close()
print(len(AF), len(JP), int((JP.atencao != '').sum()))
