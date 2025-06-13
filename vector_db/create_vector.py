import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from neo4j import GraphDatabase

load_dotenv()

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

timestamp = datetime.datetime.isoformat()

# --- Ana yükleme fonksiyonu ---
def upload_karar_with_relations(tx, karar, embedding_vector):
    # Ana karar metni
    text = (
        f"Olay Özeti: {karar.get('olay_ozeti_degerlendirme', '')}\n"
        f"Hukuki Değerlendirme: {karar.get('hukuki_nitelendirme', '')}\n"
        f"Hüküm: {karar.get('hukum', '')}"
    )

    tx.run("""
    MERGE (k:Karar {dosya_no: $dosya_no})
    SET k.text = $text,
        k.mahkeme = $mahkeme,
        k.karar_no = $karar_no,
        k.hukum = $hukum,
        k.upload_time = $timestamp
    WITH k
    CALL db.create.setNodeVectorProperty(k, "embedding", $embedding)
    RETURN k
    """, {
        "dosya_no": karar.get("dosya_no"),
        "karar_no": karar.get("karar_no"),
        "mahkeme": karar.get("mahkeme"),
        "text": text,
        "hukum": karar.get("hukum"),
        "embedding": embedding_vector,
        "timestamp": timestamp
    })

    # İlişkisel alanlar ve ilişkileri
    if karar.get("nitelikli_hal"):
        tx.run("""
        MERGE (q:Qualifier {name: $name})
        WITH q
        MATCH (k:Karar {dosya_no: $dosya_no})
        MERGE (k)-[:HAS_QUALIFIER]->(q)
        """, {"name": karar["nitelikli_hal"], "dosya_no": karar["dosya_no"]})

    if karar.get("hafifletici_sebep"):
        tx.run("""
        MERGE (h:HafifleticiSebep {name: $name})
        WITH h
        MATCH (k:Karar {dosya_no: $dosya_no})
        MERGE (k)-[:HAS_MITIGATOR]->(h)
        """, {"name": karar["hafifletici_sebep"], "dosya_no": karar["dosya_no"]})

    if karar.get("madde"):
        tx.run("""
        MERGE (m:Madde {numara: $numara})
        WITH m
        MATCH (k:Karar {dosya_no: $dosya_no})
        MERGE (k)-[:ABOUT_ARTICLE]->(m)
        """, {"numara": karar["madde"], "dosya_no": karar["dosya_no"]})

    if karar.get("sanik"):
        tx.run("""
        MERGE (s:Sanik {name: $name})
        WITH s
        MATCH (k:Karar {dosya_no: $dosya_no})
        MERGE (k)-[:HAS_DEFENDANT]->(s)
        """, {"name": karar["sanik"], "dosya_no": karar["dosya_no"]})

    if karar.get("maktul"):
        tx.run("""
        MERGE (m:Maktul {name: $name})
        WITH m
        MATCH (k:Karar {dosya_no: $dosya_no})
        MERGE (k)-[:HAS_VICTIM]->(m)
        """, {"name": karar["maktul"], "dosya_no": karar["dosya_no"]})

# --- Embedding + Upload işlemi ---
with open("decisions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            karar = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Atlanan satır: {e}")
            continue

        # Embedding’i oluştur
        combined_text = (
            f"Olay Özeti: {karar.get('olay_ozeti_degerlendirme', '')}\n"
            f"Hukuki Değerlendirme: {karar.get('hukuki_nitelendirme', '')}\n"
            f"Hüküm: {karar.get('hukum', '')}"
        )
        embedding_vector = embedding_model.embed_query(combined_text)

        # Neo4j'e yaz
        with driver.session(database="neo4j") as session:
            session.execute_write(upload_karar_with_relations, karar, embedding_vector)

print("Tüm kararlar ve ilişkili node’lar başarıyla yüklendi.")
