"""Estilo de jogo v2 (distância dos arremessos, AST%, defesa) x uso x pace x histórico x lesões."""
import pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.formula.api as smf
from scipy import stats

ps, *_ = pd.read_pickle('lesoes/estilo.pkl')          # base de jogador-temporada (>=20 jogos, >=15 min) com lesões por categoria/parte
d0, *_ = pd.read_pickle('lesoes/estilo_cruzado.pkl')  # posição, pace_z, uso_z, hist3, perdidos
bb = pd.read_pickle('lesoes/bbref_estilo.pkl')
d = d0.drop(columns=['estilo', 'grupo']).merge(bb[['personId', 'season', 'avg_dist', 'pct_fga_00_03', 'pct_fga_dunk', 'pct_fga_fg3a', 'ast_pct', 'usg_pct',
                                                    'stl_pct', 'blk_pct', 'drb_pct', 'orb_pct', 'dbpm']], on=['personId', 'season'], how='inner')
FEAT = ['avg_dist', 'pct_fga_00_03', 'pct_fga_dunk', 'pct_fga_fg3a', 'ast_pct', 'usg_pct', 'stl_pct', 'blk_pct', 'drb_pct', 'dbpm']
d = d.dropna(subset=FEAT).copy()
Z = d.groupby('season')[FEAT].transform(lambda s: (s - s.mean()) / s.std())
sil = {k: silhouette_score(Z.values, KMeans(k, n_init=10, random_state=0).fit_predict(Z.values), sample_size=3000, random_state=0) for k in range(4, 9)}
print('silhueta por k:', {k: round(v, 3) for k, v in sil.items()})
K = 6
km = KMeans(K, n_init=30, random_state=0).fit(Z.values); d['grupo'] = km.labels_
cent = pd.DataFrame(km.cluster_centers_, columns=FEAT).round(2)
print(cent.to_string())
pd.to_pickle((d, Z, cent), 'lesoes/estilo_v2_tmp.pkl')
