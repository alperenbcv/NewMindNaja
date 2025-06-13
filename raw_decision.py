from langchain.schema import Document
from neo4j import GraphDatabase
from chatbot.vector import retriever
from chatbot.graph import graph         # Neo4j driver burada zaten var

def _fetch_full_json(dosya_no: str) -> str | None:
    with graph.session(database="neo4j") as s:
        rec = s.run(
            "MATCH (k:Karar {dosya_no:$id}) RETURN k.json AS json LIMIT 1",
            {"id": dosya_no},
        ).single()
        return rec and rec["json"]

def raw_decision(query: str, k: int = 1) -> str:
    """
    Sorguyla en alakalı k kararın TAM JSON metnini döndürür.
    """
    docs: list[Document] = retriever.get_relevant_documents(query)
    if not docs:
        return "Uygun karar bulunamadı."

    outputs = []
    for d in docs[:k]:
        dosya_no = d.metadata.get("dosya_no")  # upload sırasında eklemiştik
        raw_json = _fetch_full_json(dosya_no) if dosya_no else None
        outputs.append(raw_json or d.page_content)

    return "\n\n---\n\n".join(outputs)
