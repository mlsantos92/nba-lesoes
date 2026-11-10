import pandas as pd, numpy as np
x=pd.read_pickle('dados/players_corrigido_x10_saida.pkl').copy()   # x10 já corrigido
t=pd.read_pickle('dados/teams_raw.pkl')
n0=len(x)
# 1) pré-temporada fora
x['gameType']=x.gameType.replace({'NBA Emirates Cup':'Emirates NBA Cup'})
pre=x.gameType.isin(['Preseason','Pre Season']); x=x[~pre].copy()
print('pré-temporada removida',pre.sum(),' restam',len(x))
# 2) nomes: texto UTF-8 lido como Windows-1252
def conserta(v):
    if not isinstance(v,str) or not any(ch in v for ch in 'ÃÂÄÅ'): return v
    for enc in ('cp1252','latin-1'):
        try: return v.encode(enc).decode('utf-8')
        except Exception: pass
    # caracteres que o cp1252 não tem (ex.: \x90): byte a byte
    b=bytearray()
    for ch in v:
        try: b+=ch.encode('cp1252')
        except Exception: b+=bytes([ord(ch)]) if ord(ch)<256 else ch.encode('utf-8')
    try: return b.decode('utf-8')
    except Exception: return v
antes=x.firstName.astype(str)+' '+x.lastName.astype(str)
for c in ['firstName','lastName']: x[c]=x[c].map(conserta)
mud=(x.firstName.astype(str)+' '+x.lastName.astype(str))!=antes
print('nomes consertados',mud.sum(),sorted(set((x.firstName+' '+x.lastName)[mud])))
# 3) vitória e mando no formato único 1/0
for c in ['win','home']: x[c]=x[c].map({'Yes':1,'No':0,1:1,0:0}).astype('Int64')
# 4) duplicidades (linha inteira idêntica, fora o bloco)
cols=[c for c in x.columns if c not in ('blocoJogo','corrigidoX10')]
dup=x.duplicated(subset=cols); print('duplicadas removidas',dup.sum()); x=x[~dup].copy()
# 5) percentuais recalculados das cestas: fração com 4 casas (= XX,XX%)
x=x.rename(columns={'threePointerPercentage':'threePointersPercentage','Coluna1':'freeThrowsPercentage'})
for pc,mk,at in [('fieldGoalsPercentage','fieldGoalsMade','fieldGoalsAttempted'),('threePointersPercentage','threePointersMade','threePointersAttempted'),('freeThrowsPercentage','freeThrowsMade','freeThrowsAttempted')]:
    x[pc]=(x[mk]/x[at]).where(x[at]>0).round(4)
x.to_pickle('dados/limpo_parcial.pkl')
