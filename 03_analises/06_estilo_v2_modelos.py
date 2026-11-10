import pandas as pd, numpy as np
import statsmodels.formula.api as smf

d, Z, cent = pd.read_pickle('lesoes/estilo_v2_tmp.pkl')
NOMES = {0: 'Defensor de perímetro', 1: 'Pivô de aro', 2: 'Interno de garrafão', 3: 'Pontuador de média distância', 4: 'Armador criador', 5: 'Arremessador de 3'}
d['estilo'] = d.grupo.map(NOMES)
cats = ['Ligamento', 'Muscular', 'Fratura', 'Contusão (batidas)']
partes = ['Joelho', 'Tornozelo', 'Costas', 'Pé', 'Posterior da coxa', 'Panturrilha', 'Quadril', 'Mão / dedo', 'Ombro', 'Virilha']
d['min1000'] = d['min'] / 1000
for c in Z.columns: d[c + '_z'] = Z[c]
print('jogador-temporadas:', len(d), '| jogadores:', d.personId.nunique())
print(d.groupby('estilo')[['avg_dist', 'pct_fga_00_03', 'pct_fga_dunk', 'ast_pct', 'stl_pct', 'blk_pct', 'dbpm', 'uso']].mean().round(3).to_string())
print(pd.crosstab(d.posicao, d.estilo).to_string())

def idx(g, base_c, base_p):
    return pd.concat([(g[cats].sum() / g.min1000.sum()) / base_c, (g[partes].sum() / g.min1000.sum()) / base_p])
bc = d[cats].sum() / d.min1000.sum(); bp = d[partes].sum() / d.min1000.sum()
T = d.groupby('estilo').apply(lambda g: pd.concat([pd.Series({'n': len(g), 'lesoes_1000min': g.lesoes.sum() / g.min1000.sum()}), idx(g, bc, bp)]), include_groups=False)
print('\nÍndice vs média (1 = média)\n', T.round(2).to_string())

print('\n== Mesma posição, estilos diferentes (lesões por 1.000 min | n)')
P = d.groupby(['posicao', 'estilo']).apply(lambda g: pd.Series({'n': len(g), 'taxa': g.lesoes.sum() / g.min1000.sum(), 'hist3': g.hist3.mean(),
    **{c: (g[c].sum() / g.min1000.sum()) for c in cats}}), include_groups=False)
P = P[P.n >= 80]
for p_, g in P.groupby(level=0):
    b = d[d.posicao == p_]; bb = b[cats].sum() / b.min1000.sum()
    g2 = g.copy(); g2[cats] = g2[cats] / bb.values
    print(g2.round(2).to_string())
print('\n== Mesmo histórico, estilos diferentes (lesões por 1.000 min)')
H = d.groupby(['faixaHist', 'estilo'], observed=True).apply(lambda g: pd.Series({'n': len(g), 'taxa': g.lesoes.sum() / g.min1000.sum()}), include_groups=False)
print(H.taxa.unstack().round(2).to_string()); print(H.n.unstack().astype(int).to_string())

ref = 'Treatment("Arremessador de 3")'
m = smf.poisson(f'lesoes ~ C(estilo, {ref}) + uso_z + pace_z + hist3 + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
k = [x for x in m.params.index if not x.startswith('C(season') and x != 'Intercept']
print('\n== Modelo único (razão de taxas)'); print(pd.DataFrame({'RR': np.exp(m.params[k]), 'p': m.pvalues[k]}).round(3).to_string())
for cat in cats:
    mm = smf.poisson(f'Q("{cat}") ~ C(estilo, {ref}) + uso_z + pace_z + hist3 + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
    kk = [x for x in mm.params.index if 'estilo' in x]
    print('--', cat, ' '.join(f"{x.split('T.')[1][:-1]}={np.exp(mm.params[x]):.2f}{'*' if mm.pvalues[x] < 0.05 else ''}" for x in kk))
mi = smf.poisson('lesoes ~ C(estilo) * hist3 + uso_z + pace_z + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
print('\n== estilo x histórico'); print(pd.DataFrame({'RR': np.exp(mi.params[[x for x in mi.params.index if ':hist3' in x]]), 'p': mi.pvalues[[x for x in mi.params.index if ':hist3' in x]]}).round(3).to_string())
for v in ['uso_z', 'pace_z']:
    mx = smf.poisson(f'lesoes ~ C(estilo) * {v} + hist3 + {"pace_z" if v == "uso_z" else "uso_z"} + C(posicao) + C(season)', data=d, offset=np.log(d.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': d.personId})
    kk = [x for x in mx.params.index if ':' + v in x]; print(f'\n== estilo x {v}'); print(pd.DataFrame({'RR': np.exp(mx.params[kk]), 'p': mx.pvalues[kk]}).round(3).to_string())

# características individuais (cada uma em desvios-padrão da temporada)
F = ['avg_dist_z', 'pct_fga_00_03_z', 'pct_fga_dunk_z', 'ast_pct_z', 'stl_pct_z', 'blk_pct_z', 'drb_pct_z', 'dbpm_z', 'usg_pct_z']
res = []
for alvo in ['lesoes'] + cats + partes:
    dd = d.rename(columns={alvo: 'y'})
    mm = smf.poisson('y ~ ' + ' + '.join(F) + ' + pace_z + hist3 + C(posicao) + C(season)', data=dd, offset=np.log(dd.min1000)).fit(disp=0, cov_type='cluster', cov_kwds={'groups': dd.personId})
    for f in F + ['pace_z', 'hist3']: res.append((alvo, f, np.exp(mm.params[f]), mm.pvalues[f]))
R = pd.DataFrame(res, columns=['lesao', 'caracteristica', 'RR', 'p'])
print('\n== Características (p<0,01)'); print(R[R.p < 0.01].sort_values(['lesao', 'RR']).round(3).to_string(index=False))
pd.to_pickle((d, T, P, H, R, m), 'lesoes/estilo_v2.pkl')
