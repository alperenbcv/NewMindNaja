from graph import graph

def cypher_qa(query: str) -> str:
    try:
        result = graph.query(query)
        if not result:
            return "Uygun veri bulunamadı."
        return str(result)
    except Exception as e:
        return f"Cypher sorgusunda hata: {e}"
