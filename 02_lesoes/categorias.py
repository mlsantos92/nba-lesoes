"""Categoria médica do afastamento a partir do texto do motivo (Pro Sports Transactions)."""
import re

def categoria(nota, tipo):
    """Devolve (categoria, detalhe)."""
    if tipo == 'Doença': return 'Doença', 'Doença'
    if tipo != 'Lesão': return tipo, ''
    n = str(nota).lower()
    # fratura primeiro: "fractured", "broken", "stress fracture/reaction"
    if re.search(r'fractur|broken|break\b|stress reaction', n): return 'Fratura', 'Fratura por estresse' if 'stress' in n else 'Fratura'
    # ligamentos: entorse, rompimento de ligamento, luxação
    # tendão de Aquiles e menisco contam como lesões ligamentares (decisão do estudo)
    if re.search(r'achilles', n): return 'Ligamento', 'Tendão de Aquiles'
    if re.search(r'menisc', n): return 'Ligamento', 'Menisco'
    if re.search(r'\b(acl|mcl|pcl|lcl|ucl)\b|ligament|cruciate|collateral', n):
        return 'Ligamento', 'Ruptura de ligamento' if re.search(r'torn|tear|ruptur', n) else 'Lesão de ligamento'
    if re.search(r'sprain', n): return 'Ligamento', 'Entorse'
    if re.search(r'dislocat|subluxat|separated', n): return 'Ligamento', 'Luxação'
    # tendões e cartilagem ficam em outras lesões (antes de "torn", para "torn Achilles" não virar muscular)
    if re.search(r'tendon|tendin|patellar|plantar|fasci', n): return 'Ligamento', 'Tendinite / fáscia (desgaste)'
    if re.search(r'cartilage|labrum|labral', n): return 'Ligamento', 'Cartilagem / labrum (desgaste)'
    # musculares: estiramento, contratura, espasmo, câimbra, rompimento muscular
    if re.search(r'strain|pull|spasm|tight|cramp|muscle|torn (left |right )?(hamstring|calf|groin|quad|adductor|oblique|pectoral|abdominal)', n):
        return 'Muscular', 'Estiramento / contratura'
    if re.search(r'hamstring|calf|groin|quad|adductor|oblique|abdominal', n) and re.search(r'injur|torn|tear|sore', n):
        return 'Muscular', 'Lesão muscular (sem detalhe)'
    if re.search(r'concussion', n): return 'Contusão (batidas)', 'Concussão'
    if re.search(r'bruise|contusion', n): return 'Contusão (batidas)', 'Pancada / contusão'
    if re.search(r'sore|soreness|pain|inflam|bursitis|swell|irritat', n): return 'Contusão (batidas)', 'Dor / inflamação'
    return 'Lesão sem diagnóstico', 'Só a região informada'
