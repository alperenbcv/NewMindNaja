# cli_assistant.py
# ──────────────────────────────────────────────────────────────────
"""
Basit terminal arayüzlü hukuk-KG asistanı.
$ python cli_assistant.py
"""

from chatbot.qa_chain import simple_qa
from chatbot.llm import llm
from chatbot.graph import graph
from chatbot.vector import get_similar_karar_by_embedding
from chatbot.cypher import cypher_qa
from utils import get_session_id

from langchain_neo4j import Neo4jChatMessageHistory
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory

# ─────────────────────── Araç Tanımları ──────────────────────────
tools = [
    Tool.from_function(
        name="Similar Decision Search",
        description="Verilen olay detaylarına göre vektör index’ten benzer karar(lar) getirir.",
        func=get_similar_karar_by_embedding,
    ),
    Tool.from_function(
        name="Cypher DB Search",
        description="Neo4j üzerinde Cypher sorgusu çalıştırır (risk, oran vb.).",
        func=cypher_qa,
    ),
    Tool.from_function(
        name="Direct QA Chain Search",
        description="Sadece embedding’e dayalı hızlı karar araması yapar.",
        func=simple_qa,
    ),
]

tool_names = ", ".join(t.name for t in tools)
tool_descriptions = "\n".join(f"{t.name}: {t.description}" for t in tools)

# ─────────────────────── Bellek (chat-history) ───────────────────
def get_memory(session_id: str):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# ───────────────────── ReAct Prompt - kısaltılmış şema ───────────
REACT_PREFIX = """
Sen bir hukuk karar destek sistemisin. Cevapları sadece sana verilen araçlar üzerinden üret.

GRAPH CHEATSHEET (Concise)
───────────────────────────────────────────────
Suspect {id, prior_convictions, juvenile_convictions,
         model_recidivism_probability, sentence_amount}
Recidivism {value}                    # "0","1","2"
ModelRecidivismPrediction {value}     # "0","1","2"
Karar {text, dosya_no, karar_no, mahkeme, hukum, embedding}

(Suspect)-[:HAS_RECIDIVISM]->(Recidivism)
(Suspect)-[:HAS_RECIDIVISM_PREDICTION]->(ModelRecidivismPrediction)
(Suspect)-[:COMMITTED]->(IntentionalKilling)
(Suspect)-[:HAS_PREMEDITATED_KILL]->(PremeditatedKill)
(Suspect)-[:HAS_UNJUST_PROVOCATION_MODERATE]->(UnjustProvocationModerate)

TOOLS:
------
{tools}

Tool kullanımı:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action Geçmiş konuşmalardan:
{chat_history}

Yanıtı bitirirken mutlaka şu formatı kullan:
Thought: I now know the answer
Final Answer: <kısa cevabın>

Yeni giriş: {input}
{agent_scratchpad}
""".strip()

agent_prompt = (
    PromptTemplate.from_template(REACT_PREFIX)
    .partial(tools=tool_descriptions, tool_names=tool_names)
)

# ─────────────────────── Ajan Kurulumu ───────────────────────────
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,            # konsolda zincir adımlarını görmenizi sağlar
    handle_parsing_errors=True,
    max_iterations=4,
    early_stopping_method="generate"
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ─────────────────────── Yardımcı Fonksiyon ─────────────────────
def generate_response(user_text: str) -> str:
    """Tek seferlik çağrı – terminal oturumu için rastgele session id."""
    session_id = get_session_id()
    result = chat_agent.invoke(
        {"input": user_text},
        {"configurable": {"session_id": session_id}},
    )
    return result.get("output", str(result))

# ─────────────────────── CLI Döngüsü ────────────────────────────
if __name__ == "__main__":
    print("✦ NAJA CLI – çıkmak için 'exit' yazın\n")
    while True:
        user_input = input("Soru ➜ ")
        if user_input.lower().strip() == "exit":
            break

        mode = input("Mod seç (1=Agent / 2=QA Chain) ➜ ").strip()
        if mode == "2":
            # doğrudan QAChain – araç gerekmiyor
            answer = simple_qa(user_input)
        else:
            answer = generate_response(user_input)

        print("\nYanıt:\n" + answer + "\n" + "─" * 60 + "\n")