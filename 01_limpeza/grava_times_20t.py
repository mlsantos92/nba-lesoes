import pandas as pd, numpy as np, xlsxwriter, math
t=pd.read_pickle('dados/times_20t_lesoes.pkl'); log,corr,camp=pd.read_pickle('dados/times_log_v2.pkl'); camp=camp[camp.Temporada>='2006-07']; corr=corr[corr.gameId.isin(set(t.gameId))].reset_index(drop=True); m=pd.read_pickle('dados/oficial_mud.pkl')
mt=m[m.tabela=='times']
for r in mt.itertuples(): corr.loc[len(corr)]=[r.gameId,r.time,r.coluna,r.antes,r.depois,'valor do box score oficial (Basketball-Reference); jogadores já somavam o oficial']
log=log+[('Base recortada para as últimas 20 temporadas','2006-07 a 2025-26','temporadas 1996-97 a 2005-06 retiradas'),('Jogadores afastados por jogo','2 colunas','afastados (quantidade) e afastadosLista (jogador e motivo) em cada jogo do time; detalhes em NBA_Lesoes_2006_2026.xlsx'),('Avançadas dos 5 jogos incluídos calculadas pelo box','10 linhas / 19 colunas','arremesso, distribuição de pontos, taxa de lances livres, quatro fatores do adversário e ratings/ritmo estimados; só fórmulas que reproduzem a fonte em 90% ou mais dos jogos de 2024-25 (quase todas 99-100%)'),('Totais do time corrigidos pelo box score oficial',f'{len(mt)} valores / {mt.gameId.nunique()} jogos','em todos os outros jogos com divergência o total do time já estava certo e o erro era de jogador (corrigido na tabela de jogadores); agora time e jogadores fecham em 100% da base')]
AUS=[('gameId','Data','Visitante','Mandante','Placar','Data original','Motivo do adiamento','Fonte usada'),
(22400524,'19/02/2025','Hornets','Lakers','100 x 97','09/01/2025','Incêndios em Los Angeles','Basketball-Reference; gameId confirmado no NBA.com'),
(22400532,'28/01/2025','Rockets','Hawks','100 x 96','11/01/2025','Tempestade de inverno em Atlanta','NBA.com, súmula oficial'),
(22400537,'16/03/2025','Hornets','Clippers','88 x 123','11/01/2025','Incêndios em Los Angeles','NBA.com, súmula oficial'),
(22400538,'17/03/2025','Spurs','Lakers','109 x 125','11/01/2025','Incêndios em Los Angeles','NBA.com, súmula oficial'),
(22400617,'06/04/2025','Bucks','Pelicans','111 x 107','22/01/2025','Nevasca em Nova Orleans','NBA.com, súmula oficial')]
PEND=[('#','Gravidade','Problema','Linhas / jogos','Onde','Sugestão'),
(1,'Informativo','Colunas avançadas vazias nos jogos mais antigos e nos 5 jogos incluídos',f'{int(t.q1Points.isna().sum())} linhas sem quartos','q*, ratings, percentuais avançados','A fonte não tem esses dados')]
VALID=[('Achado','Por que não é erro'),
('Perdas de bola (turnovers) com regra diferente por temporada (com ou sem as perdas coletivas)','Desconsiderado: fora do escopo da análise'),
('otAllPoints maior que ot1Points + ot2Points (87 linhas)','A fonte só tem colunas para 2 prorrogações; otAllPoints soma todas'),
('numMinutes acima de 240','240 + 25 por prorrogação; todos os valores fecham'),
('Celtics e Pacers com 81 jogos em 2012-13','Jogo cancelado após o atentado na Maratona de Boston'),
('Temporadas com menos jogos (1998-99, 2011-12, 2019-20, 2020-21)','Locaute e pandemia'),
('Campanha de 2024-25','Com os 5 jogos incluídos, os 30 times fecham com a campanha oficial'),
('Time x soma dos jogadores','Pontos, arremessos, rebotes, assistências, roubos, tocos e faltas fecham em 100% dos jogos')]
wb=xlsxwriter.Workbook('/mnt/user-data/outputs/TeamStatistics_2006_2026.xlsx',{'constant_memory':True})
fp=wb.add_format({'num_format':'0.00%'}); fd=wb.add_format({'num_format':'dd/mm/yyyy hh:mm'}); fdd=wb.add_format({'num_format':'dd/mm/yyyy'}); fh=wb.add_format({'bold':True}); fw=wb.add_format({'text_wrap':True,'valign':'top'})
PCT={'fieldGoalsPercentage','threePointersPercentage','freeThrowsPercentage'}
ws=wb.add_worksheet('Times'); cols=list(t.columns); ws.write_row(0,0,cols,fh); ws.freeze_panes(1,0)
ip={i for i,c in enumerate(cols) if c in PCT}; idt=cols.index('gameDateTimeEst'); idd=cols.index('gameDate')
for i,row in enumerate(t.itertuples(index=False,name=None),start=1):
    for k,x in enumerate(row):
        if x is None or x is pd.NA or x is pd.NaT or (isinstance(x,float) and math.isnan(x)): continue
        if k==idt: ws.write_datetime(i,k,x.to_pydatetime(),fd); continue
        if k==idd: ws.write_datetime(i,k,x.to_pydatetime(),fdd); continue
        if isinstance(x,np.integer): x=int(x)
        if isinstance(x,(float,np.floating)):
            x=float(x)
            if k in ip: ws.write_number(i,k,x,fp); continue
            if x.is_integer(): x=int(x)
        ws.write(i,k,x)
cw=wb.add_worksheet('Campanhas'); cw.write_row(0,0,list(camp.columns),fh); cw.freeze_panes(1,0)
for i,l in enumerate(camp.itertuples(index=False,name=None),1):
    for k,x in enumerate(l):
        if isinstance(x,np.integer): x=int(x)
        if k==5: cw.write_number(i,k,float(x),fp)
        else: cw.write(i,k,x)
cw.set_column(0,1,14); cw.set_column(2,8,13)
a=wb.add_worksheet('Jogos ausentes (incluídos)')
for i,l in enumerate(AUS): a.write_row(i,0,l,fh if i==0 else None)
a.set_column(0,4,12); a.set_column(5,6,26); a.set_column(7,7,50)
r=wb.add_worksheet('Correções'); r.write_row(0,0,('Etapa','Linhas afetadas','Observação'),fh)
for i,l in enumerate(log,1): r.write_row(i,0,l)
r.set_column(0,0,58); r.set_column(1,1,24); r.set_column(2,2,100)
c=wb.add_worksheet('Correções linha a linha'); c.write_row(0,0,list(corr.columns),fh)
for i,l in enumerate(corr.itertuples(index=False,name=None),1): c.write_row(i,0,[int(v) if isinstance(v,np.integer) else (float(v) if isinstance(v,np.floating) else v) for v in l])
c.set_column(0,1,14); c.set_column(5,5,90)
p=wb.add_worksheet('Erros pendentes')
for i,l in enumerate(PEND): p.write_row(i,0,l,fh if i==0 else fw)
p.set_column(0,0,4); p.set_column(1,1,12); p.set_column(2,2,70); p.set_column(3,3,22); p.set_column(4,4,24); p.set_column(5,5,52)
v=wb.add_worksheet('Validados (não são erro)')
for i,l in enumerate(VALID): v.write_row(i,0,l,fh if i==0 else fw)
v.set_column(0,1,70)
wb.close(); print('ok')