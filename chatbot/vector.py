from chatbot.llm import embedding_model
from chatbot.graph import graph
from langchain_neo4j import Neo4jVector

vs = Neo4jVector(
    embedding=embedding_model,
    graph=graph,
    index_name="kararVector",
    node_label="Karar",
    text_node_property="text",
    embedding_node_property="embedding"
)

retriever = vs.as_retriever()

def get_similar_karar_by_embedding(query: str) -> str:
    docs = retriever.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])
