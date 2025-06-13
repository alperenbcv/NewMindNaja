# chatbot/raw_decision.py
from typing import List
from langchain.schema import Document
from chatbot.vector import retriever          # VectorStoreRetriever
from chatbot.graph import graph               # Neo4jGraph (LangChain)

def _fetch_full_json(dosya_no: str) -> str | None:
    """
    Verilen dosya numarasına ait Karar düğümünün k.json alanını döndürür.
    """
    res = graph.query(
        """
        MATCH (k:Karar {dosya_no: $id})
        RETURN k.json AS json
        LIMIT 1
        """,
        params={"id": dosya_no},
    )
    return res and res[0].get("json")

def raw_decision(query: str, k: int = 1) -> str:
    """
    Sorguyla en alakalı k kararın **tam JSON** metnini döndürür.
    """
    docs: List[Document] = retriever.get_relevant_documents(query)
    if not docs:
        return "Uygun karar bulunamadı."

    outputs = []
    for d in docs[:k]:
        # Neo4jVectorStore, düğümün tüm property’lerini metadata’ya ekler
        dosya_no = d.metadata.get("dosya_no") or d.metadata.get("dosya_no_")
        raw_json = _fetch_full_json(dosya_no) if dosya_no else None
        outputs.append(raw_json or d.page_content)

    return "\n\n---\n\n".join(outputs)
