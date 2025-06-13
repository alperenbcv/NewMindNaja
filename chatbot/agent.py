from chatbot.qa_chain import simple_qa
from chatbot.llm import llm
from chatbot.graph import graph
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import Tool
from utils import get_session_id

from vector import get_similar_karar_by_embedding
from cypher import cypher_qa

tools = [
    Tool.from_function(
        name="Similar Decision Search",
        description="Verilen olay detaylarına göre Index Search yapar ve DB'den benzer kararları getirir.",
        func=get_similar_karar_by_embedding,
    ),
    Tool.from_function(
        name="Cypher DB Search",
        description="Risk, oran gibi verilere erişmek için Cypher sorgusu çalıştırır.",
        func=cypher_qa,
    ),
    Tool.from_function(
        name="Direct QA Chain Search",
        description="Sadece embedding'e dayalı hızlı ve sade karar araması yapar. QA Chain kullanır.",
        func=simple_qa,
    )
]

tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
Sen bir hukuk karar destek sistemisin. Cevapları sadece sana verilen araçlar üzerinden üret.
GRAPH CHEATSHEET (Concise)

Nodes ───────────────────────────────────────────────
Suspect {
  id,                       # STRING – benzersiz
  prior_convictions,        # INT
  juvenile_convictions,     # INT
  model_recidivism_probability,   # FLOAT (0-100 arası %)
  sentence_amount           # INT (yıl)
}

Recidivism { value }                # "0" Low · "1" Med · "2" High
ModelRecidivismPrediction { value } # "0","1","2"

# Karar (yargı kararı) dokümanları
Karar { text, dosya_no, karar_no, mahkeme, hukum, embedding }

# Sık kullanılan nitelikli / hafifletici düğümler
IntentionalKilling { label }        # ana suç
PremeditatedKill { active }         # örnek nitelikli hal
UnjustProvocationModerate { active }# örnek hafifletici
…                                   # diğer nadir label'lar için bkz. docs/schema.md

Relationships ──────────────────────────────────────
(Suspect)-[:HAS_RECIDIVISM]->(Recidivism)
(Suspect)-[:HAS_RECIDIVISM_PREDICTION]->(ModelRecidivismPrediction)

# Bağlantılar (kısa liste)
(Suspect)-[:COMMITTED]->(IntentionalKilling)
(Suspect)-[:HAS_PREMEDITATED_KILL]->(PremeditatedKill)
(Suspect)-[:HAS_UNJUST_PROVOCATION_MODERATE]->(UnjustProvocationModerate)
(Suspect)-[:HAS_DISCRETIONARY_MITIGATION]->(DiscretionaryMitigation)
…                                   # diğer nitelikli / hafifletici ilişkiler

# Karar bağlantıları
(Karar)-[:HAS_DEFENDANT]->(Sanik)
(Karar)-[:HAS_VICTIM]->(Maktul)
(Karar)-[:HAS_QUALIFIER]->(Qualifier)


TOOLS:
------
{tools}

Tool kullanımı:
```
"Thought: Do I need to use a tool? Yes\n"
    "Action: the action to take, should be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "```\n"
"Geçmiş konuşmalardan:\n{chat_history}\n\n"
    "Yeni giriş: {input}\n{agent_scratchpad}"
""").partial(
    tools=tool_descriptions,
    tool_names=tool_names
)

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},
    )
    return response["output"]


while True:
    user_input = input("Soru (çıkmak için 'exit'): ")
    if user_input.lower() == "exit":
        break

    mode = input("Kullanım modu seçin → [1] Agent tabanlı  [2] QA Chain: ")
    try:
        if mode.strip() == "1":
            answer = generate_response(user_input)
        elif mode.strip() == "2":
            from qa_chain import simple_qa
            answer = simple_qa(user_input)
        else:
            print("Lütfen sadece 1 veya 2 girin.")
            continue

        print(f"Cevap:\n{answer}\n")
    except Exception as e:
        print(f"Hata: {e}")
