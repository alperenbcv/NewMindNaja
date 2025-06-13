import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from chatbot.graph import graph

# Renk haritası (label → hex)
LABEL_COLOR = {
    "Suspect": "#007BFF",
    "ModelRecidivismPrediction": "#FF7733",
    "AgeGroup": "#8E44AD",
    "Gender": "#16A085",
    "IntentionalKilling": "#C0392B",
    # ... diğer label’leri ekleyin
}

def build_rich_graph(suspect_ids: list[str]):
    data = graph.query(CYPHER_TEMPLATE, {"suspectIds": suspect_ids})
    if not data:
        return None

    G = nx.MultiDiGraph()
    rec = data[0]  # Tek satır
    for node in rec["nodes"]:
        lbl = list(node.labels)[0]
        G.add_node(
            node.id,
            label=lbl if lbl != "Suspect" else node.get("name", "Suspect"),
            title=f"{lbl}<br/>{node._properties}",
            color=LABEL_COLOR.get(lbl, "#AAAAAA"),
            shape="dot" if lbl == "Suspect" else "ellipse",
        )
    for rel in rec["rels"]:
        G.add_edge(
            rel.start_node.id,
            rel.end_node.id,
            label=rel.type,
            title=rel.type,
            color="#CCCC66" if rel.type.startswith("HAS") else "#888888",
        )
    return G

def similar_suspects_graph(risk_class: int, prob_pct: float, k=3, tol_pct=5.0):
    # --- 1. Benzer sanıkları bul (önceki kod) -----------
    low, high = max(prob_pct - tol_pct, 0.0), min(prob_pct + tol_pct, 100.0)
    cy = (
        "MATCH (s:Suspect)-[:HAS_RECIDIVISM_PREDICTION]->"
        "(m:ModelRecidivismPrediction {value:$risk}) "
        "WHERE s.model_recidivism_probability >= $low "
        "AND   s.model_recidivism_probability <= $high "
        "RETURN s.id AS id "
        "ORDER BY abs(s.model_recidivism_probability-$prob) "
        "LIMIT $k"
    )
    rows = graph.query(cy, {"risk": str(risk_class), "low": low,
                            "high": high, "prob": prob_pct, "k": k})
    suspect_ids = [r["id"] for r in rows]
    if not suspect_ids:
        components.html("<p>Benzer sanık bulunamadı.</p>", height=80); return

    # --- 2. Alt-grafı getir ---
    G = build_rich_graph(suspect_ids)
    if G is None:
        components.html("<p>Graf veri yok.</p>", height=80); return

    # --- 3. Görselleştir ---
    net = Network(height="550px", directed=False)
    net.from_nx(G)
    net.repulsion(node_distance=250, spring_length=250)
    components.html(net.generate_html(), height=550, scrolling=False)
