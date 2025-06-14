from chatbot.vector import retriever
from chatbot.llm import llm
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def simple_qa(question: str) -> str:
    result = qa_chain.invoke({"query": question})
    return result["result"]
