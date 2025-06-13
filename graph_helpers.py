# ---------- graph_helpers.py ----------
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from chatbot.graph import graph           # Neo4jGraph örneğiniz

def similar_suspects_graph(
    risk_class: int,        # 0 / 1 / 2
    prob_pct: float,        # 0.0 – 100.0 arası olasılık (%)
    k: int = 3,
    tol_pct: float = 5.0    # ±5 puan tolerans
) -> None:
    """
    Aynı recidivism sınıfında olup olasılığı
    `prob_pct ± tol_pct` aralığında kalan k sanığı getirir
    ve grafiği çizer.
    """
    low  = max(prob_pct - tol_pct, 0.0)
    high = min(prob_pct + tol_pct, 100.0)

    cypher = """
    MATCH (s:Suspect)
      WHERE s.model_recidivism_probability >= $low
        AND s.model_recidivism_probability <= $high
    MATCH (s)-[:HAS_RECIDIVISM_PREDICTION]->(m:ModelRecidivismPrediction {value:$risk})
    RETURN s.id   AS id,
           coalesce(s.name,'Suspect') AS label,
           s.model_recidivism_probability AS p
    ORDER BY abs(s.model_recidivism_probability - $prob)
    LIMIT $k
    """
    recs = graph.query(
        cypher,
        low=low, high=high,
        risk=str(risk_class),
        prob=prob_pct, k=k
    )

    if not recs:
        components.html("<p>Benzer sanık bulunamadı.</p>", height=80)
        return

    # ----- Graf -----
    G = nx.Graph()
    G.add_node("Current", label=f"Current ({prob_pct:.1f}%)", color="#FFCC00")

    for r in recs:
        lbl = f"{r['label']} ({r['p']:.1f}%)"
        G.add_node(r["id"], label=lbl, color="#007BFF")
        G.add_edge("Current", r["id"], title=f"{r['p']:.1f}%")

    net = Network(height="450px", directed=False)
    net.from_nx(G)
    net.repulsion(node_distance=220, spring_length=220)

    components.html(net.generate_html(), height=450, scrolling=False)
