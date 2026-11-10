"""Fórmulas das estatísticas avançadas (padrão NBA) calculadas a partir do box score."""
import pandas as pd, numpy as np

def div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = a / b
    return np.where(b == 0, np.nan, r)

def posses(fga, fta, oreb, opp_dreb, fgm, tov):
    """Posses de um time pela fórmula de Dean Oliver (usada pela NBA nas estimativas)."""
    return fga + 0.4 * fta - 1.07 * div(oreb, oreb + opp_dreb) * (fga - fgm) + tov

def times(t):
    """Avançadas por linha de time; usa a linha do adversário do mesmo jogo."""
    o = t.set_index(['gameId', 'teamId'])
    k = list(zip(t.gameId, t.opponentTeamId))
    O = o.loc[k].reset_index(drop=True); O.index = t.index
    F = lambda c: t[c].astype(float); G = lambda c: O[c].astype(float)
    tov_tot = F('turnovers') + F('turnoversTeam').fillna(0) * 0  # perdas como na coluna da fonte
    p1 = posses(F('fieldGoalsAttempted'), F('freeThrowsAttempted'), F('reboundsOffensive'), G('reboundsDefensive'), F('fieldGoalsMade'), F('turnovers'))
    p2 = posses(G('fieldGoalsAttempted'), G('freeThrowsAttempted'), G('reboundsOffensive'), F('reboundsDefensive'), G('fieldGoalsMade'), G('turnovers'))
    poss = 0.5 * (p1 + p2)
    mins = F('numMinutes')
    r = pd.DataFrame(index=t.index)
    r['possessions'] = poss
    r['offensiveRating'] = 100 * div(F('teamScore'), poss)
    r['defensiveRating'] = 100 * div(G('teamScore'), poss)
    r['netRating'] = r.offensiveRating - r.defensiveRating
    r['pace'] = 48 * div(poss, mins / 5)
    r['pacePer40'] = 40 * div(poss, mins / 5)
    r['effectiveFieldGoalPercentage'] = div(F('fieldGoalsMade') + 0.5 * F('threePointersMade'), F('fieldGoalsAttempted'))
    r['trueShootingPercentage'] = div(F('teamScore'), 2 * (F('fieldGoalsAttempted') + 0.44 * F('freeThrowsAttempted')))
    r['assistPercentage'] = div(F('assists'), F('fieldGoalsMade'))
    r['assistToTurnoverRatio'] = div(F('assists'), F('turnovers'))
    r['assistRatio'] = 100 * div(F('assists'), F('fieldGoalsAttempted') + 0.44 * F('freeThrowsAttempted') + F('assists') + F('turnovers'))
    r['teamTurnoverPercentage'] = div(F('turnovers'), poss)
    r['offensiveReboundPercentage'] = div(F('reboundsOffensive'), F('reboundsOffensive') + G('reboundsDefensive'))
    r['defensiveReboundPercentage'] = div(F('reboundsDefensive'), F('reboundsDefensive') + G('reboundsOffensive'))
    r['reboundPercentage'] = div(F('reboundsTotal'), F('reboundsTotal') + G('reboundsTotal'))
    r['freeThrowAttemptRate'] = div(F('freeThrowsAttempted'), F('fieldGoalsAttempted'))
    r['opponentEffectiveFieldGoalPercentage'] = div(G('fieldGoalsMade') + 0.5 * G('threePointersMade'), G('fieldGoalsAttempted'))
    r['opponentFreeThrowAttemptRate'] = div(G('freeThrowsAttempted'), G('fieldGoalsAttempted'))
    r['opponentTurnoverPercentage'] = div(G('turnovers'), poss)
    r['opponentOffensiveReboundPercentage'] = 1 - r['defensiveReboundPercentage']
    pts = F('teamScore'); fg2 = F('fieldGoalsMade') - F('threePointersMade')
    r['percentFieldGoalAttempts2Point'] = div(F('fieldGoalsAttempted') - F('threePointersAttempted'), F('fieldGoalsAttempted'))
    r['percentFieldGoalAttempts3Point'] = div(F('threePointersAttempted'), F('fieldGoalsAttempted'))
    r['percentPoints2Point'] = div(2 * fg2, pts)
    r['percentPoints3Point'] = div(3 * F('threePointersMade'), pts)
    r['percentPointsFreeThrow'] = div(F('freeThrowsMade'), pts)
    r['percentPointsFastBreak'] = div(F('pointsFastBreak'), pts)
    r['percentPointsOffTurnovers'] = div(F('pointsFromTurnovers'), pts)
    r['percentPointsInPaint'] = div(F('pointsInThePaint'), pts)
    r['percentPoints2PointMidRange'] = div(2 * fg2 - F('pointsInThePaint'), pts)
    gm = lambda X: (X('teamScore') + X('fieldGoalsMade') + X('freeThrowsMade') - X('fieldGoalsAttempted') - X('freeThrowsAttempted') + X('reboundsDefensive')
                    + 0.5 * X('reboundsOffensive') + X('assists') + X('steals') + 0.5 * X('blocks') - X('foulsPersonal') - X('turnovers'))
    r['playerImpactEstimate'] = div(gm(F), gm(F) + gm(G))
    return r

def jogadores(j, t):
    """Avançadas por jogador que só dependem do box do jogador e dos totais do time/adversário."""
    tt = t.set_index(['gameId', 'teamName'])
    T = tt.loc[list(zip(j.gameId, j.playerteamName))].reset_index(drop=True); T.index = j.index
    tt2 = t.set_index(['gameId', 'teamId'])
    O = tt2.loc[list(zip(T.gameId if 'gameId' in T else j.gameId, T.opponentTeamId))].reset_index(drop=True); O.index = j.index
    P = lambda c: j[c].astype(float); Tm = lambda c: T[c].astype(float); Op = lambda c: O[c].astype(float)
    mn = P('numMinutes'); tmin = Tm('numMinutes')
    r = pd.DataFrame(index=j.index)
    r['effectiveFieldGoalPercentage'] = div(P('fieldGoalsMade') + 0.5 * P('threePointersMade'), P('fieldGoalsAttempted'))
    r['trueShootingPercentage'] = div(P('points'), 2 * (P('fieldGoalsAttempted') + 0.44 * P('freeThrowsAttempted')))
    r['assistToTurnoverRatio'] = div(P('assists'), P('turnovers'))
    r['assistRatio'] = 100 * div(P('assists'), P('fieldGoalsAttempted') + 0.44 * P('freeThrowsAttempted') + P('assists') + P('turnovers'))
    r['assistPercentage'] = div(P('assists'), div(mn, tmin / 5) * Tm('fieldGoalsMade') - P('fieldGoalsMade'))
    r['offensiveReboundPercentage'] = div(P('reboundsOffensive') * (tmin / 5), mn * (Tm('reboundsOffensive') + Op('reboundsDefensive')))
    r['defensiveReboundPercentage'] = div(P('reboundsDefensive') * (tmin / 5), mn * (Tm('reboundsDefensive') + Op('reboundsOffensive')))
    r['reboundPercentage'] = div(P('reboundsTotal') * (tmin / 5), mn * (Tm('reboundsTotal') + Op('reboundsTotal')))
    r['usagePercentage'] = div((P('fieldGoalsAttempted') + 0.44 * P('freeThrowsAttempted') + P('turnovers')) * (tmin / 5),
                               mn * (Tm('fieldGoalsAttempted') + 0.44 * Tm('freeThrowsAttempted') + Tm('turnovers')))
    r['estimatedUsagePercentage'] = r['usagePercentage']
    r['teamTurnoverPercentage'] = div(P('turnovers'), P('fieldGoalsAttempted') + 0.44 * P('freeThrowsAttempted') + P('turnovers'))
    r['estimatedTurnoverPercentage'] = r['teamTurnoverPercentage']
    pts = P('points'); fg2 = P('fieldGoalsMade') - P('threePointersMade')
    r['percentFieldGoalAttempts2Point'] = div(P('fieldGoalsAttempted') - P('threePointersAttempted'), P('fieldGoalsAttempted'))
    r['percentFieldGoalAttempts3Point'] = div(P('threePointersAttempted'), P('fieldGoalsAttempted'))
    r['percentPoints2Point'] = div(2 * fg2, pts)
    r['percentPoints3Point'] = div(3 * P('threePointersMade'), pts)
    r['percentPointsFreeThrow'] = div(P('freeThrowsMade'), pts)
    for col, a, b in [('FieldGoalsMade', 'fieldGoalsMade', 'fieldGoalsMade'), ('FieldGoalsAttempted', 'fieldGoalsAttempted', 'fieldGoalsAttempted'),
                      ('ThreePointersMade', 'threePointersMade', 'threePointersMade'), ('ThreePointersAttempted', 'threePointersAttempted', 'threePointersAttempted'),
                      ('FreeThrowsMade', 'freeThrowsMade', 'freeThrowsMade'), ('FreeThrowsAttempted', 'freeThrowsAttempted', 'freeThrowsAttempted'),
                      ('OffensiveRebounds', 'reboundsOffensive', 'reboundsOffensive'), ('DefensiveRebounds', 'reboundsDefensive', 'reboundsDefensive'),
                      ('Rebounds', 'reboundsTotal', 'reboundsTotal'), ('Assists', 'assists', 'assists'), ('Turnovers', 'turnovers', 'turnovers'),
                      ('Steals', 'steals', 'steals'), ('Blocks', 'blocks', 'blocks'), ('FoulsPersonal', 'foulsPersonal', 'foulsPersonal'),
                      ('Points', 'points', 'teamScore')]:
        r['percentTeam' + col] = div(P(a), Tm(b))
    gm = lambda X, pts_col: (X(pts_col) + X('fieldGoalsMade') + X('freeThrowsMade') - X('fieldGoalsAttempted') - X('freeThrowsAttempted') + X('reboundsDefensive')
                             + 0.5 * X('reboundsOffensive') + X('assists') + X('steals') + 0.5 * X('blocks') - X('foulsPersonal') - X('turnovers'))
    r['playerImpactEstimate'] = div(gm(P, 'points'), gm(Tm, 'teamScore') + gm(Op, 'teamScore'))
    return r
