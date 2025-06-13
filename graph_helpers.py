# ---------- graph_helpers.py ----------
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from chatbot.graph import graph

CYPHER_TEMPLATE = """
UNWIND $suspectIds AS sid
MATCH  (s:Suspect {id: sid})
OPTIONAL MATCH (s)-[r1]->(n1)
OPTIONAL MATCH (n1)-[r2]->(n2)
WITH  collect(distinct s)  AS Ss,
      collect(distinct n1) AS Ns1,
      collect(distinct n2) AS Ns2,
      collect(distinct r1) + collect(distinct r2) AS Rs
RETURN
  [n IN Ss+Ns1+Ns2 |
     { id:   elementId(n),
       labels: labels(n),
       props: properties(n) }
  ] AS nodes,
  [r IN Rs |
     { s: elementId(startNode(r)),
       e: elementId(endNode(r)),
       type: type(r) }
  ] AS rels
   """

LABEL_COLOR = {
    "Suspect": "#007BFF",
    "ModelRecidivismPrediction": "#FF7733",
    "AgeGroup": "#8E44AD",
    "Gender": "#16A085",
    "IntentionalKilling": "#C0392B",
    "Recidivism": "#D35400",
}

def build_rich_graph(suspect_ids):
    recs = graph.query(CYPHER_TEMPLATE, {"suspectIds": suspect_ids})
    if not recs:
        return None
    data = recs[0]
    G = nx.MultiDiGraph()
    for n in data["nodes"]:
        labels = n["labels"]
        label0 = labels[0] if labels else "Node"
        props  = n["props"]
        G.add_node(
            n["id"],
            label=props.get("name", label0) if "Suspect" in labels else label0,
            title=f"{label0}<br/>" + "<br/>".join(f"{k}: {v}" for k, v in props.items()),
            color=LABEL_COLOR.get(label0, "#AAAAAA"),
            shape="dot" if "Suspect" in labels else "ellipse",
        )
    for r in data["rels"]:
        G.add_edge(
            r["s"], r["e"],
            label=r["type"],
            title=r["type"],
            color="#CCCC66" if r["type"].startswith("HAS") else "#888888",
        )
    return G

def similar_suspects_graph(risk_class, prob_pct, k=3, tol_pct=5.0):
    low, high = max(prob_pct - tol_pct, 0.0), min(prob_pct + tol_pct, 100.0)
    cy = (
        "MATCH (s:Suspect)-[:HAS_RECIDIVISM_PREDICTION]->"
        "(m:ModelRecidivismPrediction {value:$risk}) "
        "WHERE s.model_recidivism_probability >= $low "
        "  AND s.model_recidivism_probability <= $high "
        "RETURN s.id AS id "
        "ORDER BY abs(s.model_recidivism_probability - $prob) "
        "LIMIT $k"
    )
    ids = [r["id"] for r in graph.query(cy,
           {"risk": str(risk_class), "low": low, "high": high,
            "prob": prob_pct, "k": k})]
    if not ids:
        components.html("<p>Benzer sanık bulunamadı.</p>", height=80); return

    G = build_rich_graph(ids)
    if G is None:
        components.html("<p>Grafik verisi yok.</p>", height=80); return

    net = Network(height="550px", directed=False, bgcolor="#ffffff")
    net.from_nx(G)
    net.repulsion(node_distance=240, spring_length=240)
    components.html(net.generate_html(), height=550, scrolling=False)
