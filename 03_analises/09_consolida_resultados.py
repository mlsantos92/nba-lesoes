"""Junta os resultados de todas as análises de lesões em uma planilha de estudo e em um JSON para a página do estudo."""
import pandas as pd, numpy as np, json
import statsmodels.formula.api as smf

OUT = 'estudo/NBA_Lesoes_Estudo_Resultados.xlsx'
LES = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Lesão sem diagnóstico']
cats = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)']
abas, graf = {}, {}
fit = lambda f, d, g: smf.ols(f, data=d).fit(cov_type='cluster', cov_kwds={'groups': d[g]})
def coefs(m, keep, exp=False):
    ci = m.conf_int()
    f = (lambda x: np.exp(x)) if exp else (lambda x: x)
    return pd.DataFrame({'efeito': f(m.params[keep]), 'ic95_inf': f(ci.loc[keep, 0]), 'ic95_sup': f(ci.loc[keep, 1]), 'p_valor': m.pvalues[keep]}).round(4)

# ---------------- base de lesões
E, M = pd.read_pickle('lesoes/episodios.pkl')
L = E[E.categoria.isin(LES) & (E.jogosPerdidos > 0)]
c1 = L.groupby('categoria').agg(afastamentos=('afastamentoId', 'size'), jogosPerdidos=('jogosPerdidos', 'sum'), medianaJogos=('jogosPerdidos', 'median')).sort_values('afastamentos', ascending=False)
c2 = L[L.parteDoCorpo != ''].groupby('parteDoCorpo').agg(afastamentos=('afastamentoId', 'size'), jogosPerdidos=('jogosPerdidos', 'sum'), medianaJogos=('jogosPerdidos', 'median')).sort_values('afastamentos', ascending=False).head(15)
c3 = L.groupby('temporada').agg(afastamentos=('afastamentoId', 'size'), jogosPerdidos=('jogosPerdidos', 'sum'))
abas['2 Lesões - visão geral'] = [('Afastamentos por categoria (com pelo menos 1 jogo perdido)', c1), ('Partes do corpo mais afetadas', c2), ('Afastamentos por temporada', c3)]
graf['porTemporada'] = {'temporadas': c3.index.tolist(), 'afastamentos': c3.afastamentos.tolist()}
graf['categorias'] = {'nomes': c1.index.tolist(), 'afastamentos': c1.afastamentos.tolist(), 'jogos': c1.jogosPerdidos.astype(int).tolist()}

# ---------------- uso x pace (jogadores com uso > 20%)
b = pd.read_pickle('lesoes/uso_pace_base.pkl')
f1 = b.groupby('faixaUso', observed=True).agg(jogadorTemporadas=('personId', 'size'), pctComLesao=('teveLesao', 'mean'), jogosPerdidosMedia=('jogosPerdidosLesao', 'mean'), pctJogosPerdidos=('taxaPerdida', 'mean'), minutosPorJogo=('mpj', 'mean')).round(3)
f3 = (b.pivot_table(index='faixaUso', columns='faixaPace', values='taxaPerdida', aggfunc='mean', observed=True) * 100).round(1)
m1 = fit('taxa100 ~ uso100 + pace + mpj + C(season)', b, 'personId')
ti = {s: i for i, s in enumerate(sorted(b.season.unique()))}
Lall = L.assign(ti=L.temporada.map(ti)).groupby(['personId', 'ti']).size()
b['ti'] = b.season.map(ti); b['lesoes3anos'] = [sum(Lall.get((p, k), 0) for k in range(i - 3, i)) for p, i in zip(b.personId, b.ti)]
b2 = b[b.ti >= 3]
m2 = fit('taxa100 ~ uso100 + pace + mpj + lesoes3anos + C(season)', b2, 'personId')
reg = pd.concat({'Sem histórico': coefs(m1, ['uso100', 'pace', 'mpj']), 'Com histórico (lesões nas 3 temporadas anteriores)': coefs(m2, ['uso100', 'pace', 'mpj', 'lesoes3anos'])})
abas['3 Uso e pace'] = [('Jogadores com uso > 20%: faixas de uso', f1), ('% de jogos perdidos por lesão: uso x ritmo do time (terços dentro da temporada)', f3),
                        ('OLS: pontos percentuais de jogos perdidos (efeito fixo de temporada, erro agrupado por jogador)', reg)]
graf['uso'] = {'faixas': f1.index.astype(str).tolist(), 'pct': (f1.pctJogosPerdidos * 100).round(1).tolist()}
graf['usoCoef'] = {'sem': round(m1.params['uso100'], 3), 'p_sem': round(m1.pvalues['uso100'], 4), 'com': round(m2.params['uso100'], 3), 'p_com': round(m2.pvalues['uso100'], 3), 'hist': round(m2.params['lesoes3anos'], 3)}

# ---------------- recorrência
rec = pd.read_pickle('lesoes/recorrencia.pkl')
rec = rec.assign(risco_sem_hist=lambda x: (x.risco_sem_hist * 100).round(1), risco_com_hist=lambda x: (x.risco_com_hist * 100).round(1), risco_relativo=lambda x: x.risco_relativo.round(2))
rec = rec[['risco_sem_hist', 'risco_com_hist', 'risco_relativo', 'n_sem', 'n_com']].rename(columns={'risco_sem_hist': '% com lesão no local (sem histórico)', 'risco_com_hist': '% com lesão no local (com histórico)'})
abas['4 Recorrência'] = [('Risco de lesão na mesma parte do corpo, com e sem lesão nela nas 3 temporadas anteriores', rec),
                         ('Resumo', pd.DataFrame({'valor': ['4,2%', '16,1%', '3,79', '2,11 (p < 0,001)', '67% (1.032 de 1.535)']},
                                                  index=['Risco sem histórico (todas as partes)', 'Risco com histórico', 'Risco relativo bruto', 'Odds ratio controlando minutos, parte e temporada', 'Recorrências no mesmo lado']))]
graf['recorrencia'] = {'partes': rec.index.tolist(), 'rr': rec.risco_relativo.tolist()}

# ---------------- pace do time
T = pd.read_pickle('lesoes/time_pace.pkl')
from scipy import stats
cor = pd.DataFrame([(a, bb, *stats.spearmanr(T[a], T[bb])) for a, bb in [('pace', 'perdidosPorJogo'), ('pace', 'lesoesPor82'), ('pace_z', 'perdidosPorJogo_z'), ('pace_z', 'lesoesPor82_z')]],
                   columns=['x', 'y', 'spearman', 'p_valor']).round(4)
cor['leitura'] = ['bruto: ligado à tendência histórica', 'bruto: ligado à tendência histórica', 'dentro da temporada', 'dentro da temporada']
ter = T.groupby('terco', observed=True)[['pace', 'perdidosPorJogo', 'lesoesPor82']].mean().round(2)
abas['5 Pace do time'] = [('Correlação ritmo x lesões do elenco (600 times-temporada)', cor), ('Terços de ritmo dentro de cada temporada', ter)]
tt = T.groupby('season')[['pace', 'lesoesPor82']].mean()
graf['paceTempo'] = {'temporadas': tt.index.tolist(), 'pace': tt.pace.round(1).tolist(), 'lesoes': tt.lesoesPor82.round(1).tolist()}
graf['paceZ'] = [[round(x, 2), round(y, 2)] for x, y in zip(T.pace_z, T.lesoesPor82_z)]
graf['paceCor'] = {'bruto': round(stats.spearmanr(T.pace, T.lesoesPor82)[0], 2), 'z': round(stats.spearmanr(T.pace_z, T.lesoesPor82_z)[0], 2)}

# ---------------- estilos v2
d, Tst, P, H, R, m = pd.read_pickle('lesoes/estilo_v2.pkl')
_, Z, cent = pd.read_pickle('lesoes/estilo_v2_tmp.pkl')
NOMES = {0: 'Defensor de perímetro', 1: 'Pivô de aro', 2: 'Interno de garrafão', 3: 'Pontuador de média distância', 4: 'Armador criador', 5: 'Arremessador de 3'}
cent.index = [NOMES[i] for i in cent.index]
cent.columns = ['dist. média', '% arremessos 0-3 pés', '% enterradas', '% de 3', 'AST%', 'USG%', 'STL%', 'BLK%', 'DRB%', 'DBPM']
perfil = d.groupby('estilo').agg(jogadorTemporadas=('personId', 'size'), distMedia_pes=('avg_dist', 'mean'), pctAro=('pct_fga_00_03', 'mean'), pctEnterradas=('pct_fga_dunk', 'mean'),
                                 pct3=('pct_fga_fg3a', 'mean'), astPct=('ast_pct', 'mean'), usgPct=('usg_pct', 'mean'), stlPct=('stl_pct', 'mean'), blkPct=('blk_pct', 'mean'), drbPct=('drb_pct', 'mean'), dbpm=('dbpm', 'mean')).round(3)
sil = pd.DataFrame({'k': [4, 5, 6, 7, 8], 'silhueta': [0.201, 0.199, 0.156, 0.166, 0.163]}).set_index('k')
ex = d.sort_values('min', ascending=False).drop_duplicates(['estilo', 'personId']).groupby('estilo').head(4).groupby('estilo').jogador.apply(lambda s: ', '.join(s.str.replace('_', ' ')))
abas['6 Estilos de jogo'] = [('Centróides (z dentro da temporada)', cent), ('Perfil médio de cada estilo', perfil), ('Silhueta por k (escolhido k = 6 pela interpretação)', sil),
                             ('Estilo x posição (jogador-temporadas)', pd.crosstab(d.posicao, d.estilo)), ('Exemplos (jogador-temporadas com mais minutos)', ex.to_frame('exemplos'))]
abas['7 Estilo x tipo de lesão'] = [('Índice vs média da liga (1 = média), por 1.000 minutos', Tst.round(2))]
P2 = P.copy()
abas['8 Mesma posição e histórico'] = [('Mesma posição, estilos diferentes (taxa = lesões por 1.000 min; categorias = índice vs média da posição)', P2.round(2)),
                                     ('Mesmo histórico, estilos diferentes: lesões por 1.000 min', H.taxa.unstack().round(2)), ('Mesmo histórico: jogador-temporadas', H.n.unstack())]
k = [x for x in m.params.index if not x.startswith('C(season') and x != 'Intercept']
mod = coefs(m, k, exp=True); mod.index = [x.replace('C(estilo, Treatment("Arremessador de 3"))[T.', 'Estilo: ').replace('C(posicao)[T.', 'Posição: ').rstrip(']') for x in mod.index]
ref = 'Treatment("Arremessador de 3")'
porcat = {}
for c in cats:
    mm = smf.poisson(f'Q("{c}") ~ C(estilo, {ref}) + uso_z + pace_z + hist3 + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
    kk = [x for x in mm.params.index if 'estilo' in x]
    porcat[c] = pd.Series(np.exp(mm.params[kk]).round(2).astype(str).values + np.where(mm.pvalues[kk] < 0.05, '*', ''), index=[x.split('T.')[1][:-1] for x in kk])
mi = smf.poisson('lesoes ~ C(estilo, ' + ref + ') * hist3 + uso_z + pace_z + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
ki = [x for x in mi.params.index if ':hist3' in x]; inter = coefs(mi, ki, exp=True); inter.index = [x.split('T.')[1].split(']')[0] + ' × histórico' for x in ki]
abas['9 Modelo de estilo'] = [('Poisson: lesões por minuto (razão de taxas; referência = Arremessador de 3; efeito fixo de temporada; erro agrupado por jogador)', mod),
                             ('Razão de taxas por categoria (* = p < 0,05)', pd.DataFrame(porcat)), ('Interação estilo × lesões anteriores (quanto cada lesão anterior pesa a mais que no Arremessador de 3)', inter)]
graf['estiloRR'] = {'nomes': [x.replace('Estilo: ', '') for x in mod.index if x.startswith('Estilo')], 'rr': mod.loc[[x for x in mod.index if x.startswith('Estilo')], 'efeito'].round(2).tolist(),
                    'inf': mod.loc[[x for x in mod.index if x.startswith('Estilo')], 'ic95_inf'].round(2).tolist(), 'sup': mod.loc[[x for x in mod.index if x.startswith('Estilo')], 'ic95_sup'].round(2).tolist()}
graf['estiloIdx'] = Tst[['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)', 'Joelho', 'Tornozelo', 'Pé', 'Posterior da coxa', 'Ombro', 'Costas']].round(2).reset_index().to_dict('records')
graf['estiloHist'] = H.taxa.unstack().round(2).reset_index().to_dict('records')
graf['modelo'] = {x: [round(mod.loc[x, 'efeito'], 3), round(mod.loc[x, 'p_valor'], 4)] for x in ['hist3', 'uso_z', 'pace_z']}
NOMEF = {'avg_dist_z': 'Distância média', 'pct_fga_00_03_z': '% arremessos no aro', 'pct_fga_dunk_z': '% enterradas', 'ast_pct_z': 'AST%', 'stl_pct_z': 'STL%', 'blk_pct_z': 'BLK%',
         'drb_pct_z': 'DRB%', 'dbpm_z': 'DBPM', 'usg_pct_z': 'USG%', 'pace_z': 'Pace do time', 'hist3': 'Lesões nas 3 temporadas anteriores'}
R2 = R.assign(caracteristica=R.caracteristica.map(NOMEF), RR=R.RR.round(3), p=R.p.round(4))
abas['10 Características'] = [('Razão de taxas por +1 desvio-padrão (todas juntas no modelo, + posição e temporada)', R2.pivot(index='caracteristica', columns='lesao', values='RR')),
                              ('p-valores', R2.pivot(index='caracteristica', columns='lesao', values='p'))]

# ---------------- jogo da lesão
r, T1, o1, o2, o3, o4, oc = pd.read_pickle('lesoes/jogo_lesao_res.pkl')
taxa = lambda s: round(s.mean() * 1000, 2)
ant = r.groupby('faixaAnt', observed=True).lesao.agg(jogos='size', lesoesPor1000=taxa)
j7 = r.groupby(r.jogos7d.clip(upper=4)).lesao.agg(jogos='size', lesoesPor1000=taxa); j7.index = ['0', '1', '2', '3', '4+']
pes = pd.concat({'> 1 DP': r.groupby('pesado1dp_7d').lesao.agg(jogos='size', lesoesPor1000=taxa), '> 2 DP': r.groupby('pesado2dp_7d').lesao.agg(jogos='size', lesoesPor1000=taxa)})
NOMEV = {'adv_contato': 'Defesa física do adversário (z)', 'pesado1dp_7d': 'Jogo > 1 DP de minutos nos 7 dias anteriores', 'pesado2dp_7d': 'Jogo > 2 DP de minutos nos 7 dias anteriores',
         'jogos7d': 'Jogos nos 7 dias anteriores', 'b2b': 'Segundo jogo em dias seguidos', 'casa': 'Em casa', 'prorrogacao': 'Jogo com prorrogação', 'adv_faltas100': 'Faltas do adversário por 100 posses (z)',
         'adv_ftrCedido': 'Lances livres cedidos pelo adversário (z)', 'adv_tocos100': 'Tocos do adversário por 100 posses (z)', 'zMax7d': 'Maior z de minutos nos 7 dias anteriores',
         'faltasAdvJogo100': 'Faltas do adversário no jogo por 100 posses', 'faltasSofridas36': 'Faltas sofridas pelo jogador por 36 min'}
ren = lambda o: o.rename(index=NOMEV).rename(columns={'OR': 'odds ratio', 'p': 'p_valor'})
ocp = oc.reset_index().rename(columns={'index': 'variavel'}); ocp['variavel'] = ocp.variavel.map(NOMEV)
abas['11 Jogo da lesão'] = [('Lesões por 1.000 jogos por quinto da defesa física do adversário', T1),
                            ('Lesões por 1.000 jogos conforme os minutos do jogo anterior (z na temporada do jogador)', ant), ('Jogo pesado nos 7 dias anteriores', pes), ('Jogos nos 7 dias anteriores', j7),
                            ('Logit condicional (mesmo jogador-temporada): modelo com > 1 DP', ren(o1)), ('Modelo com > 2 DP', ren(o2)), ('Componentes da defesa física', ren(o3)),
                            ('Contato no próprio jogo', ren(o4)), ('Por categoria (odds ratio)', ocp.pivot(index='variavel', columns='categoria', values='OR').round(3)),
                            ('Por categoria (p-valor)', ocp.pivot(index='variavel', columns='categoria', values='p').round(4))]
graf['jogoAnt'] = {'faixas': ant.index.astype(str).tolist(), 'taxa': ant.lesoesPor1000.tolist()}
graf['jogoOR'] = {k2: [round(o1.loc[k2, 'OR'], 3), round(o1.loc[k2, 'ic95_inf'], 3), round(o1.loc[k2, 'ic95_sup'], 3)] for k2 in o1.index}
graf['jogoOR']['pesado2dp_7d'] = [round(o2.loc['pesado2dp_7d', 'OR'], 3), round(o2.loc['pesado2dp_7d', 'ic95_inf'], 3), round(o2.loc['pesado2dp_7d', 'ic95_sup'], 3)]
graf['contatoCat'] = {c: [round(oc[(oc.categoria == c)].loc['adv_contato', 'OR'], 3), round(oc[(oc.categoria == c)].loc['adv_contato', 'p'], 4),
                          round(oc[(oc.categoria == c)].loc['pesado1dp_7d', 'OR'], 3), round(oc[(oc.categoria == c)].loc['pesado1dp_7d', 'p'], 4)] for c in cats}
graf['jogoN'] = {'jogos': int(len(r)), 'lesoes': int(r.lesao.sum())}

# ---------------- leia-me, dados e metodologia
leia = pd.DataFrame({'conteudo': [
    'Estudo: o que explica as lesões na NBA? Estilo de jogo, carga de minutos, histórico, uso, ritmo do time e defesa física do adversário.',
    'Período: temporadas 2006-07 a 2025-26 (20 temporadas), temporada regular.',
    'Fontes: box scores de jogadores e times (Kaggle, NBA), corrigidos com súmulas oficiais e Basketball-Reference; transações de lista de lesionados da Pro Sports Transactions (3 cópias públicas); estatísticas de arremesso e avançadas do Basketball-Reference.',
    'Unidades de análise: afastamento (episódio de lesão), jogador-temporada e jogador-jogo.',
    'Cada aba traz uma tabela de resultados; a aba Metodologia descreve como cada número foi obtido.']}, index=['O que é', 'Período', 'Fontes', 'Unidades', 'Como ler'])
dados = pd.DataFrame({'valor': ['786.814 linhas, 38.023 jogos, 2.859 jogadores (30 temporadas)', '76.046 linhas (30 temporadas)', '539.311 linhas, 25.746 jogos, 2.169 jogadores',
                                '35.180 registros', '18.434 afastamentos / 122.500 jogos perdidos', '10.659 lesões / 81.010 jogos perdidos', '12.897 jogador-temporadas, 99,85% ligadas',
                                '146 valores de jogadores e 11 de times corrigidos; 7 jogos de jogadores e 5 de times incluídos', 'Soma dos jogadores = total do time em 100% dos jogos',
                                '36 de 40 jogos perdidos sorteados confirmados no Basketball-Reference; casos conhecidos (Rose, Durant, Wall, Kawhi, George, Kobe) conferidos']},
                     index=['Box score de jogadores (limpo)', 'Box score de times (limpo)', 'Recorte de 20 temporadas', 'Registros de lista de lesionados (2006-2026)', 'Afastamentos montados',
                            'Afastamentos por lesão física (com jogo perdido)', 'Estilo (Basketball-Reference)', 'Correções', 'Consistência', 'Validação dos afastamentos'])
abas['1 Dados e limpeza'] = [('Base de dados', dados)]
met = pd.DataFrame({'como foi feito': [
    'Afastamento = colocação na lista de lesionados até a primeira volta (ativação, nova colocação ou primeiro jogo disputado). Jogos perdidos = jogos do time no intervalo. Entre temporadas só continua se o jogador estava na liga.',
    'Categorias por palavras-chave do motivo: Fratura, Ligamento (inclui Aquiles, menisco, tendinite, cartilagem), Muscular, Contusão (batidas, dores, concussão), Lesão sem diagnóstico. Doença, COVID, descanso e outros ficam fora.',
    'Uso e pace: jogadores com uso médio > 20% (≥ 20 jogos). OLS da % de jogos perdidos com efeito fixo de temporada e erro agrupado por jogador; depois com lesões das 3 temporadas anteriores.',
    'Pace do time: correlação de Spearman bruta e com z-score dentro da temporada (remove a tendência histórica comum a pace e registro de lesões).',
    'Recorrência: para cada jogador-temporada e parte do corpo, risco com e sem lesão no mesmo local nas 3 temporadas anteriores; logit controlando minutos, parte e temporada.',
    'Estilo: K-means (k = 6, 30 inicializações) sobre 10 métricas padronizadas dentro da temporada: distância média, % no aro, % enterradas, % de 3, AST%, USG%, STL%, BLK%, DRB%, DBPM. Jogador-temporadas com ≥ 20 jogos e ≥ 15 min por jogo, 2009-10 em diante.',
    'Modelo de estilo: Poisson das lesões com exposição em minutos (offset log), estilo + uso + pace + lesões nas 3 temporadas anteriores + posição + temporada; erro agrupado por jogador; interações estilo × histórico, uso e pace.',
    'Jogo da lesão: último jogo disputado até 3 dias antes da colocação na lista. Logit condicional por jogador-temporada (compara o jogo da lesão com os outros jogos do mesmo jogador). Carga = algum jogo nos 7 dias anteriores com minutos > média + 1 ou 2 desvios-padrão da temporada do jogador.',
    'Defesa física do adversário: média do z-score na temporada de faltas por 100 posses, lances livres cedidos (FTr) e tocos por 100 posses.']},
    index=['Afastamentos', 'Classificação', 'Uso e pace', 'Pace do time', 'Recorrência', 'Estilo de jogo', 'Modelo de estilo', 'Jogo da lesão', 'Defesa física'])
lim = pd.DataFrame({'limitação': ['A data de entrada na lista nem sempre é o dia da lesão; lesões de treino não se ligam a jogo.', 'Registro de lesões mudou ao longo do tempo (mais registros recentes); por isso tudo é comparado dentro da temporada.',
                                  'Muitos testes por categoria e parte do corpo: resultados isolados com p entre 0,01 e 0,05 devem ser lidos com cautela.', 'Estudo observacional: associações, não causalidade (ex.: técnicos poupam jogadores frágeis em jogos seguidos).',
                                  '28% dos afastamentos por lesão não informam o diagnóstico.']})
abas['Metodologia'] = [('Métodos', met), ('Limitações', lim)]
ordem = ['Leia-me', '1 Dados e limpeza', '2 Lesões - visão geral', '3 Uso e pace', '4 Recorrência', '5 Pace do time', '6 Estilos de jogo', '7 Estilo x tipo de lesão', '8 Mesma posição e histórico',
         '9 Modelo de estilo', '10 Características', '11 Jogo da lesão', 'Metodologia']
abas['Leia-me'] = [('Estudo de lesões na NBA (2006-07 a 2025-26)', leia)]

with pd.ExcelWriter(OUT, engine='xlsxwriter') as w:
    wb = w.book; tit = wb.add_format({'bold': True, 'font_size': 12, 'font_color': '#1F3A5F'})
    for nome in ordem:
        row = 0; ws = None
        for titulo, df in abas[nome]:
            df = df if isinstance(df, pd.DataFrame) else df.to_frame()
            df.to_excel(w, sheet_name=nome, startrow=row + 1)
            ws = w.sheets[nome]; ws.write(row, 0, titulo, tit)
            row += len(df) + df.columns.nlevels + 3 + (1 if isinstance(df.index, pd.MultiIndex) else 0)
        ws.set_column(0, 0, 42); ws.set_column(1, 40, 16)
        if nome in ('Leia-me', 'Metodologia', '1 Dados e limpeza'): ws.set_column(1, 1, 140)
json.dump(graf, open('estudo/graficos.json', 'w'), ensure_ascii=False, default=float)
print('ok', OUT)
