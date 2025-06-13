# ---------- graph_helpers.py ----------
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from chatbot.graph import graph           # <─ Neo4jGraph örneğiniz

# 1) 2-hop alt-graf şablonu
CYPHER_TEMPLATE = """
UNWIND $suspectIds AS sid
MATCH  (s:Suspect {id:sid})
OPTIONAL MATCH (s)-[r1]->(n1)
OPTIONAL MATCH (n1)-[r2]->(n2)
WITH  collect(distinct s)  AS Ss,
      collect(distinct n1) AS Ns1,
      collect(distinct n2) AS Ns2,
      collect(distinct r1) + collect(distinct r2) AS Rs
RETURN Ss + Ns1 + Ns2 AS nodes, Rs AS rels
"""

# 2) Basit renk haritası  (etiket → renk)
LABEL_COLOR = {
    "Suspect": "#007BFF",
    "ModelRecidivismPrediction": "#FF7733",
    "AgeGroup": "#8E44AD",
    "Gender": "#16A085",
    "IntentionalKilling": "#C0392B",
    "Recidivism": "#D35400",
    # gerekirse diğer label’leri ekleyin…
}

# ---------------- İç yardımcı ----------------
def build_rich_graph(suspect_ids: list[str]) -> nx.MultiDiGraph | None:
    """Verilen id’ler için alt-graf döner (None → veri yok)."""
    recs = graph.query(CYPHER_TEMPLATE, {"suspectIds": suspect_ids})
    if not recs:
        return None

    rec = recs[0]  # tek satır bekleniyor
    G = nx.MultiDiGraph()

    # Düğümler
    for node in rec["nodes"]:
        label = list(node.labels)[0]
        G.add_node(
            node.id,
            label=node.get("name", label) if label == "Suspect" else label,
            title=f"{label}<br/>{node._properties}",
            color=LABEL_COLOR.get(label, "#AAAAAA"),
            shape="dot" if label == "Suspect" else "ellipse",
        )

    # Kenarlar
    for rel in rec["rels"]:
        G.add_edge(
            rel.start_node.id,
            rel.end_node.id,
            label=rel.type,
            title=rel.type,
            color="#CCCC66" if rel.type.startswith("HAS") else "#888888",
        )
    return G

# ---------------- Ana fonksiyon ----------------
def similar_suspects_graph(
    risk_class: int,           # 0 / 1 / 2
    prob_pct: float,           # 0‒100 ölçeğinde olasılık
    k: int = 3,
    tol_pct: float = 5.0,
) -> None:
    """Benzer sanıkları bulur ve Streamlit içinde grafiği gömer."""
    low  = max(prob_pct - tol_pct, 0.0)
    high = min(prob_pct + tol_pct, 100.0)

    cy = (
        "MATCH (s:Suspect)-[:HAS_RECIDIVISM_PREDICTION]->"
        "(m:ModelRecidivismPrediction {value:$risk}) "
        "WHERE s.model_recidivism_probability >= $low "
        "  AND s.model_recidivism_probability <= $high "
        "RETURN s.id AS id "
        "ORDER BY abs(s.model_recidivism_probability - $prob) "
        "LIMIT $k"
    )
    rows = graph.query(
        cy, {"risk": str(risk_class), "low": low, "high": high,
             "prob": prob_pct, "k": k}
    )
    suspect_ids = [r["id"] for r in rows]

    if not suspect_ids:
        components.html("<p>Benzer sanık bulunamadı.</p>", height=80)
        return

    G = build_rich_graph(suspect_ids)
    if G is None:
        components.html("<p>Grafik verisi yok.</p>", height=80)
        return

    net = Network(height="550px", directed=False, bgcolor="#ffffff")
    net.from_nx(G)
    net.repulsion(node_distance=240, spring_length=240)

    components.html(net.generate_html(), height=550, scrolling=False)
