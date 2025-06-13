# ---------- graph_helpers.py (yeni yardımcı modül) ----------
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from chatbot.vector import get_similar_karar_by_embedding  # 3 benzer karar

def similar_suspects_graph(question: str, k: int = 3):
    """Soru cümlesine göre en yakın k karar içindeki sanıkları döndürür
       ve grafiği çizdirir."""
    docs_text = get_similar_karar_by_embedding(question)    # string birleşik
    # Diyelim ki her karar JSON satırı içeriyor -> sanık adlarını ayıklayalım
    import json, re
    suspects = []
    for line in docs_text.splitlines():
        try:
            d = json.loads(line)
            suspects.append(d["sanik"])
        except Exception:
            m = re.search(r'"sanik":\s*"?([^",}]+)', line)
            if m:
                suspects.append(m.group(1))
        if len(suspects) >= k:
            break

    # Oyuncak graf: merkez düğüm = "Query", k adet sanık etrafında
    G = nx.Graph()
    G.add_node("Query", label="Query")
    for s in suspects:
        G.add_node(s, label="Suspect")
        G.add_edge("Query", s)

    net = Network(height="450px", directed=False)
    net.from_nx(G)
    net.repulsion(node_distance=200)

    html = net.generate_html()
    components.html(html, height=450, scrolling=False)
