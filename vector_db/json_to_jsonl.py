import json

with open("decisions.json", "r", encoding="utf-8") as f:
    kararlar = json.load(f)

with open("decisions.jsonl", "w", encoding="utf-8") as f:
    for karar in kararlar:
        f.write(json.dumps(karar, ensure_ascii=False) + "\n")
