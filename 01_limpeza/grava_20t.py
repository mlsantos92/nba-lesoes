PEND=[('#','Gravidade','Erro','Tamanho','Origem','Como resolver'),
(1,'Alto','Box score incompleto: soma dos pontos dos jogadores ≠ placar do time (ao menos 1 caso de jogador creditado ao adversário)','27 times-jogo em 25 jogos','fonte (Kaggle)','marcar os jogos; não usar em análises de soma por time'),
(1,'Alto','Soma do +/- do time ≠ 5 × saldo do jogo','80 times-jogo','fonte (Kaggle)','marcar; não usar +/- desses jogos'),
(3,'Médio','Arquivo de times: 2 jogos com placar 0 e sem jogadores (Magic x Spurs 01/04/2025, Jazz x Wizards 19/03/2025)','2 jogos','fonte (Kaggle)','excluir ou buscar o placar real'),
(3,'Médio','60+ métricas avançadas e posição de titular perdidas','base inteira','conversão para Excel','trazer da fonte se o estudo precisar'),
(5,'Baixo','Arquivo de times: soma dos quartos ≠ placar','49 linhas','fonte (Kaggle)','marcar; usar o placar final'),
(5,'Baixo','Duplo-duplo: marcado errado / vazio','6 / 56 linhas','fonte (Kaggle)','recalcular pela regra (2 categorias com 10 ou mais)'),
(6,'Baixo','Minutos arredondados até 2023-24 (0 = menos de 30 segundos); fracionados a partir de 2024-25','base inteira','fonte (Kaggle)','considerar ao calcular estatísticas por minuto')]
VALID=[('Caso','Situação'),
('Cal Bowdler com 7 faltas (13/11/1999)','correto: erro humano no jogo, uma falta não foi registrada na hora'),
('Heat x Hawks de 19/12/2007 com Marion e Bibby pelos novos times','correto: final do jogo rejogado em 08/03/2008, depois das trocas'),
('Original sem ID do time em 90.798 linhas (2000, 2001, 2021, 2022)','resolvido nesta base pelo arquivo de times'),
('Jogadores com 0 minuto e alguma estatística (1 inconsistente + 25 suspeitos)','mantidos como jogos atípicos para o estudo; marcados na coluna alertaMinutos'),
('Jogadores com 0 minuto que entraram em quadra (+/-, posses) ou registraram 1-2 lances','mantidos: entraram nos segundos finais'),
('Temporadas 1998-99 (725 jogos), 2011-12 (990), 2012-13 (1.229), 2019-20 (1.059) e 2020-21 (1.080)','corretas: greves, jogo cancelado em Boston, pandemia e bolha'),
('Temporada 2024-25 com 1.223 de 1.230 jogos','incompleta na fonte: 5 jogos ausentes no arquivo de times e 2 com placar 0; tratar no arquivo de times')]


PEND=[PEND[0],
(1,'Informativo','Linhas dos 6 jogos montados pelas súmulas (marcadas com * em observacoes): avançadas de arremesso calculadas; sem ratings, uso, percentuais de equipe, tocos e faltas sofridos, posses e pontos por tipo','124 linhas / 6 jogos','súmula oficial não traz esses campos','deixar vazio')]
VALID=[v for v in VALID if not v[0].startswith('Temporada 2024-25') and not v[0].startswith('Jogadores com 0 minuto')]+[('Minutos','inteiros em toda a base; quem jogou menos de 30 segundos conta 1 minuto; o valor original fica em numMinutesFonte'),('Soma do +/- dos jogadores ≠ 5 × saldo do jogo (37 times-jogo) e perdas de bola do time com regra diferente por temporada','desconsiderados: fora do escopo da análise'),('Temporada 2024-25 com 1.230 jogos','completa: 5 jogos ausentes e 2 que vinham 0 x 0 incluídos pelas súmulas oficiais')]
import pandas as pd, numpy as np, xlsxwriter, math
f=pd.read_pickle('dados/jogadores_20t_lesoes.pkl')
PCT={c for c in f.columns if 'Percentage' in c or c.startswith('percent')}
wb=xlsxwriter.Workbook('/mnt/user-data/outputs/PlayerStatistics_2006_2026.xlsx',{'constant_memory':True}); wb.use_zip64()
fp=wb.add_format({'num_format':'0.00%'}); fd=wb.add_format({'num_format':'dd/mm/yyyy hh:mm'}); fdd=wb.add_format({'num_format':'dd/mm/yyyy'}); fh=wb.add_format({'bold':True}); fw=wb.add_format({'text_wrap':True,'valign':'top'})
ws=wb.add_worksheet('Jogadores'); cols=list(f.columns); ws.write_row(0,0,cols,fh); ws.freeze_panes(1,0)
ip=[i for i,c in enumerate(cols) if c in PCT]; idt=cols.index('gameDateTimeEst'); idd=cols.index('gameDate')
for i,row in enumerate(f.itertuples(index=False,name=None),start=1):
    for j,x in enumerate(row):
        if x is None or x is pd.NA or x is pd.NaT or (isinstance(x,float) and math.isnan(x)): continue
        if j==idt: ws.write_datetime(i,j,x.to_pydatetime(),fd); continue
        if j==idd: ws.write_datetime(i,j,x.to_pydatetime(),fdd); continue
        if isinstance(x,np.integer): x=int(x)
        if isinstance(x,(float,np.floating)):
            x=float(x)
            if j in ip: ws.write_number(i,j,x,fp); continue
            if x.is_integer(): x=int(x)
        ws.write(i,j,x)
r=wb.add_worksheet('Correções')
L=[('Etapa','Linhas afetadas','Observação'),
('Estatísticas ×10 divididas por 10','804.280','16 colunas; conferido 100% com o original do Kaggle'),
('Pré-temporada removida','50.604','Preseason e Pre Season'),
('All-Star removido','794','não é jogo oficial'),
('Final da NBA Cup removida','18 (1 jogo)','a final não conta na classificação'),
('Jogos da NBA Cup (grupos e mata-mata) mantidos como temporada regular','1.397','rótulo "Emirates NBA Cup" preservado em gameLabel'),
('Nomes com acento corrigidos','29','estavam em jogos de pré-temporada'),
('Duplicidades removidas','4','linhas idênticas'),
('Percentuais recalculados (XX,XX%)','todas','cestas ÷ tentativas; vazio quando não houve tentativa'),
('Vitória/Derrota e Mandante/Visitante','todas','antes misturava Yes/No e 1/0'),
('Blocos com vários jogos separados (Finais e final do Leste de 2022)','181 (8 jogos)','cada jogo ligado pelo placar exato'),
('Jogo da série preenchido nos playoffs de 2022','1.891','vazio na fonte; vem do último dígito do gameId'),
('Ligação com o arquivo de times','787.383 (100%)','gameId, data e IDs dos times; conferida com o original'),
('ID do jogador (personId) e chave de jogador (playerKey)','787.383 (100%)','do original, por jogo + nome + time; homônimos separados em playerKey'),
('Homônimos separados','12 nomes / 3.592 linhas','playerKey = nome + "(desde AAAA)", ano do 1º jogo na base'),
('Minutos jogados (numMinutes) trazidos da fonte','todas','por personId + gameId'),
('Jogadores que não entraram removidos','712','0 minuto, tudo zero, +/- 0 e nenhuma posse'),
('Jogos atípicos de minutos marcados (alertaMinutos)','26','1 inconsistente e 25 suspeitos; mantidos para o estudo'),
('Temporada (season) pelo ID do jogo','todas','evita que os jogos da bolha (jul–out/2020) caiam em 2020-21'),
('Data do jogo (gameDate), só informativa','todas','dia do jogo no horário de Nova York, sem a hora; não entra em nenhuma regra'),
('Jogos da bolha marcados em observacoes','3.596 linhas / 172 jogos','88 de classificação, 1 de play-in e 83 de playoffs; sem público e mando apenas formal')]

for e in pd.read_pickle('dados/jogadores_log_20t.pkl'): L.append(e)
for i,l in enumerate(L): r.write_row(i,0,l,fh if i==0 else None)
r.set_column(0,0,62); r.set_column(1,1,24); r.set_column(2,2,80)
p=wb.add_worksheet('Erros pendentes')
for i,l in enumerate(PEND): p.write_row(i,0,l,fh if i==0 else fw)
p.set_column(0,0,4); p.set_column(1,1,14); p.set_column(2,2,70); p.set_column(3,3,30); p.set_column(4,4,22); p.set_column(5,5,48)
v=wb.add_worksheet('Validados (não são erro)')
for i,l in enumerate(VALID): v.write_row(i,0,l,fh if i==0 else fw)
v.set_column(0,0,70); v.set_column(1,1,70)

o=wb.add_worksheet('Correções pelo box oficial'); mm=pd.read_pickle('dados/oficial_mud.pkl'); mm=mm[mm.tabela=='jogadores'].drop(columns='tabela')
o.write_row(0,0,['gameId','time','jogador','coluna','antes','depois'],fh)
for i,l in enumerate(mm.itertuples(index=False,name=None),1): o.write_row(i,0,[int(l[0]),l[1],l[2],l[3],float(l[4]),float(l[5])])
o.set_column(0,3,22)
wb.close(); print('ok')
