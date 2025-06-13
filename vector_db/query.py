from langchain_neo4j import Neo4jVector
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

retriever = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="kararVector",                      # index adın neyse onu yaz
    node_label="Karar",                         # kararları bu label ile kaydettiysen
    embedding_node_property="embedding",           # embedding'leri bu key ile kaydettiysen
    text_node_property="text"  # arama yapılacak metin alanı
)

docs = retriever.similarity_search("tasarlayarak kasten öldürme", k=5)

for i, doc in enumerate(docs):
    print(f"\n--- {i+1}. Karar ---")
    print(doc.page_content)
