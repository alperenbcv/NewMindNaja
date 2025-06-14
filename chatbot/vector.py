from langchain_neo4j import Neo4jVector
from chatbot.llm   import embedding_model
from chatbot.graph import graph

# ── VektörStore nesnesi ──────────────────────────────────────────
vs = Neo4jVector(
    embedding=embedding_model,
    graph=graph,
    index_name="kararVector",
    node_label="Karar",
    text_node_property="text",
    embedding_node_property="embedding",
)


retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},   )

def get_similar_karar_by_embedding(query: str, k: int = 3) -> str:

    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)
