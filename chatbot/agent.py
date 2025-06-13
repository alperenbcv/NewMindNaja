from qa_chain import simple_qa
from llm import llm
from graph import graph
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

# Ajan prompt'u
agent_prompt = PromptTemplate.from_template("""
Sen bir hukuk karar destek sistemisin. Cevapları sadece sana verilen araçlar üzerinden üret.
Database üzerinde bir cypher sorgusu oluştururken aşağıdaki node ve relationship'leri kullan.
model_recidivism_probability: Eğer açıkça oran, yüzde, ihtimal veya model tarafından tahmin edilen tekrar oranı soruluyorsa bu property'e git. 
ModelRecidivismPrediction: Ayrı bir node’dur, modelin sınıflandırma sonucunu ("0", "1", "2") içerir.
Suspect ile arasında şu ilişki vardır: (:Suspect)-[:HAS_RECIDIVISM_PREDICTION]->(:ModelRecidivismPrediction)
Recidivism: Ayrı bir node’dur. Gerçek (etiketli) suç tekrarını içerir. (:Suspect)-[:HAS_RECIDIVISM]->(:Recidivism) ilişkisi vardır. `value` alanı `"0"`, `"1"` veya `"2"` olabilir. Bu bir string'dir.
Eğer sadece "recidivism değeri" deniyorsa bu, `Recidivism.value` property’sini ifade eder.
Cevaplarında formatlama karakteri (örneğin ```) kullanma. 
Cypher sorgularını düz şekilde yaz ve sadece bir kez çalıştır.

Örnek bir karar formatı aşağıdaki gibidir:
{{"mahkeme": "İstanbul 13. Ağır Ceza Mahkemesi", "dosya_no": "2022/245 E.", "karar_no": "2023/612 K.", "sanik": "Okan S.", "maktul": "Sedat V.", "suç": "kasten öldürme", "madde": "TCK 82/1-a", "nitelikli_hal": ["tasarlayarak"], "ceza": "ağırlaştırılmış müebbet", "hafifletici_sebep": null, "olay_yeri": "Kadıköy/İstanbul – sokak üzeri", "silah_tipi": "tabanca", "eylem_tarzi": "iki hafta keşif yapıp pusu kurarak yakın mesafeden ateş etme", "pişmanlık": false, "savunma": "inkâr", "olay_ozeti_degerlendirme": "Sanık Okan S., maktul Sedat V. ile eski ortaklıkları sırasında yaşanan borç anlaşmazlığı nedeniyle husumet beslemiştir. Olaydan önce iki hafta boyunca maktulün iş çıkış saatlerini takip ederek güzergâh tespit etmiş, 18 Ekim 2022 gecesi dar bir ara sokakta pusuya düşürüp üç el ateş etmiştir. Kameralar, baz istasyonu verileri ve balistik inceleme ile eylem sabittir.", "hukuki_nitelendirme": "Uzun süreli takip, uygun zamanı kollama ve eylemden hemen sonra soğukkanlı kaçış, öldürme kararının önceden verildiğini gösterir. Bu nedenle TCK 82/1-a kapsamında tasarlayarak kasten öldürme suçu oluşmuştur.", "hukum": "Sanığın tasarlayarak kasten öldürme suçunu işlediği sabit görüldüğünden TCK 82/1-a gereği **ağırlaştırılmış müebbet hapis** cezasına hükmolunmuştur. Takdiri indirim uygulanmamıştır."}}


GRAPH SCHEMA:
------
Node properties:
---Suspect'in bilgilerini içeren node'lar---
Suspect {{id: STRING, prior_convictions: INTEGER, juvenile_convictions: INTEGER, model_recidivism_probability: FLOAT, sentence_amount: INTEGER}}
AgeGroup {{value: STRING}}
Gender {{value: STRING}}
Housing {{value: STRING}}
Race {{value: STRING}}
Recidivism {{value: STRING}}
PriorProbationViolation {{name: STRING, active: BOOLEAN}}
PriorIncarceration {{name: STRING, active: BOOLEAN}}
SubstanceAbuseHistory {{name: STRING, active: BOOLEAN}}
MentalHealthIssues {{name: STRING, active: BOOLEAN}}
GangAffiliation {{name: STRING, active: BOOLEAN}}
ComplianceHistory {{name: STRING, active: BOOLEAN}}
MotivationToChange {{name: STRING, active: BOOLEAN}}
PositiveSocialSupport {{name: STRING, active: BOOLEAN}}
EducationLevel {{value: STRING}}
MaritalStatus {{value: STRING}}
EmploymentStatus {{value: STRING}}
ModelRecidivismPrediction {{value: STRING}} 
HasDependents {{name: STRING, active: BOOLEAN}}
AggressionHistory {{name: STRING, active: BOOLEAN}}
StableEmployment {{name: STRING, active: BOOLEAN}}
SentenceType {{value: STRING}}
IsFixedTerm {{name: STRING, active: BOOLEAN}}
---Kasten öldürme suçunun nitelikli hallerine ait node'lar---
IntentionalKilling {{label: STRING}}
BloodFeud {{name: STRING, active: BOOLEAN}}
VictimIsRelative {{name: STRING, active: BOOLEAN}}
VictimIsChild {{name: STRING, active: BOOLEAN}}
PremeditatedKill {{name: STRING, active: BOOLEAN}}
MonstrousManner {{name: STRING, active: BOOLEAN}}
ToCoverAnotherCrime {{name: STRING, active: BOOLEAN}}
DestructiveManner {{name: STRING, active: BOOLEAN}}
VictimPublicServant {{name: STRING, active: BOOLEAN}}
Femicide {{name: STRING, active: BOOLEAN}}
Tradition {{name: STRING, active: BOOLEAN}}
FailedCrime {{name: STRING, active: BOOLEAN}}
---Kasten öldürme suçunu hafifleştirici ya da ortadan kaldıran node'lar---
UnjustProvocationSevere {{name: STRING, active: BOOLEAN}}
UnjustProvocationModerate {{name: STRING, active: BOOLEAN}}
PartialMentalDisorder {{name: STRING, active: BOOLEAN}}
UnjustProvocationMild {{name: STRING, active: BOOLEAN}}
DiscretionaryMitigation {{name: STRING, active: BOOLEAN}}
MitigationAge15_17 {{name: STRING, active: BOOLEAN}}
MitigationAge12_14 {{name: STRING, active: BOOLEAN}}
Deaf18_21 {{name: STRING, active: BOOLEAN}}
Deaf15_17 {{name: STRING, active: BOOLEAN}}
---Karar Node'unun bağlı olduğu diğer node'lar---
Karar {{embedding: LIST, text: STRING, dosya_no: STRING, karar_no: STRING, mahkeme: STRING, hukum: STRING, upload_time: STRING}}
Qualifier {{name: LIST}}
Madde {{numara: STRING}}
Sanik {{name: STRING}}
Maktul {{name: STRING}}
HafifleticiSebep {{name: LIST}}
Session {{id: STRING}}
Message {{content: STRING, role: STRING}}


Relationship properties:
The relationships:
(:Suspect)-[:HAS_EDUCATION]->(:EducationLevel)
(:Suspect)-[:HAS_MARITAL_STATUS]->(:MaritalStatus)
(:Suspect)-[:HAS_GENDER]->(:Gender)
(:Suspect)-[:HAS_HOUSING]->(:Housing)
(:Suspect)-[:HAS_RACE]->(:Race)
(:Suspect)-[:IN_AGE_GROUP]->(:AgeGroup)
(:Suspect)-[:HAS_DEPENDENTS]->(:HasDependents)
(:Suspect)-[:HAS_COMPLIANCE_HISTORY]->(:ComplianceHistory)
(:Suspect)-[:HAS_MOTIVATION_TO_CHANGE]->(:MotivationToChange)
(:Suspect)-[:HAS_POSITIVE_SOCIAL_SUPPORT]->(:PositiveSocialSupport)
(:Suspect)-[:HAS_EMPLOYMENT]->(:EmploymentStatus)
(:Suspect)-[:HAS_RECIDIVISM]->(:Recidivism)
(:Suspect)-[:HAS_RECIDIVISM_PREDICTION]->(:ModelRecidivismPrediction)
(:Suspect)-[:HAS_STABLE_EMPLOYMENT]->(:StableEmployment)
(:Suspect)-[:HAS_SENTENCE_TYPE]->(:SentenceType)
(:Suspect)-[:COMMITTED]->(:IntentionalKilling)
(:Suspect)-[:HAS_DISCRETIONARY_MITIGATION]->(:DiscretionaryMitigation)
(:Suspect)-[:HAS_PREMEDITATED_KILL]->(:PremeditatedKill)
(:Suspect)-[:HAS_VIOLATED_PROBATION]->(:PriorProbationViolation)
(:Suspect)-[:VICTIM_WAS_WOMAN]->(:Femicide)
(:Suspect)-[:HAS_IMPRISONED]->(:PriorIncarceration)
(:Suspect)-[:ABUSED_SUBSTANCE]->(:SubstanceAbuseHistory)
(:Suspect)-[:HAS_MENTAL_ISSUES]->(:MentalHealthIssues)
(:Suspect)-[:HAS_FIXED_TERM]->(:IsFixedTerm)
(:Suspect)-[:HAS_UNJUST_PROVOCATION_MODERATE]->(:UnjustProvocationModerate)
(:Suspect)-[:HAS_UNJUST_PROVOCATION_MILD]->(:UnjustProvocationMild)
(:Suspect)-[:VICTIM_WAS_CHILD]->(:VictimIsChild)
(:Suspect)-[:HAS_VICTIM_RELATION]->(:VictimIsRelative)
(:Suspect)-[:HAS_PARTIAL_MENTAL_DISORDER]->(:PartialMentalDisorder)
(:Suspect)-[:IS_BLOOD_FEUD]->(:BloodFeud)
(:Suspect)-[:IS_TRADITION_MURDER]->(:Tradition)
(:Suspect)-[:USED_MONSTROUS_MANNER]->(:MonstrousManner)
(:Suspect)-[:HAS_AGGRESSION_HISTORY]->(:AggressionHistory)
(:Suspect)-[:HAS_MOTIVE_COVER_CRIME]->(:ToCoverAnotherCrime)
(:Suspect)-[:HAS_UNJUST_PROVOCATION_SEVERE]->(:UnjustProvocationSevere)
(:Suspect)-[:HAS_GANG_AFFILIATION]->(:GangAffiliation)
(:Suspect)-[:HAS_AGE_12_14]->(:MitigationAge12_14)
(:Suspect)-[:HAS_AGE_15_17]->(:MitigationAge15_17)
(:Suspect)-[:USED_DESTRUCTIVE_MANNER]->(:DestructiveManner)
(:Suspect)-[:VICTIM_WAS_PUBLIC_SERVANT]->(:VictimPublicServant)
(:Suspect)-[:IS_DEAF_15_17]->(:Deaf15_17)
(:Suspect)-[:DUE_TO_FAILED_CRIME]->(:FailedCrime)
(:Suspect)-[:IS_DEAF_18_21]->(:Deaf18_21)
(:Karar)-[:HAS_QUALIFIER]->(:Qualifier)
(:Karar)-[:ABOUT_ARTICLE]->(:Madde)
(:Karar)-[:HAS_DEFENDANT]->(:Sanik)
(:Karar)-[:HAS_VICTIM]->(:Maktul)
(:Karar)-[:HAS_MITIGATOR]->(:HafifleticiSebep)
(:Session)-[:LAST_MESSAGE]->(:Message)
(:Message)-[:NEXT]->(:Message)


TOOLS:
------
{tools}

Tool kullanımı:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
Geçmiş konuşmalardan:
{chat_history}

Yeni giriş: {input}
{agent_scratchpad}
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
