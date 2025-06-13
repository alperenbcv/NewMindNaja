# chatbot/raw_decision.py
from typing import List
from langchain.schema import Document
from chatbot.vector import retriever            # retriever = VectorStoreRetriever

def raw_decision(query: str, k: int = 1) -> str:
    """
    Sorguyla en alakalı kararın TAM metnini döndürür.
    """
    docs: List[Document] = retriever.get_relevant_documents(query)  # ← değişiklik
    if not docs:
        return "Uygun karar bulunamadı."
    return "\n\n---\n\n".join(d.page_content for d in docs[:k])
