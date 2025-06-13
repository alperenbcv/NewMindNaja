from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory

from chatbot.qa_chain import simple_qa
from chatbot.llm import llm
from chatbot.graph import graph
from chatbot.utils import get_session_id
from chatbot.vector import get_similar_karar_by_embedding
from chatbot.cypher import cypher_qa
from fractions import Fraction
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from graph_helpers import similar_suspects_graph

load_dotenv()



TOOLS = [
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
    ),
]

_tool_names = ", ".join(t.name for t in TOOLS)
_tool_descs = "\n".join(f"{t.name}: {t.description}" for t in TOOLS)


AGENT_PROMPT = PromptTemplate.from_template(
    """
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
    """
).partial(tools=_tool_descs, tool_names=_tool_names)

_agent = create_react_agent(llm, TOOLS, AGENT_PROMPT)
_agent_executor = AgentExecutor(
    agent=_agent,
    tools=TOOLS,
    verbose=True,
    handle_parsing_errors=True,
)

chat_agent = RunnableWithMessageHistory(
    _agent_executor,
    lambda session_id: Neo4jChatMessageHistory(session_id=session_id, graph=graph),
    input_messages_key="input",
    history_messages_key="chat_history",
)


def generate_response(user_text: str, session_id: str, mode: str = "Agent") -> str:
    if mode == "Agent":
        result = chat_agent.invoke(
            {"input": user_text},
            {"configurable": {"session_id": session_id}},
        )
        return result["output"]
    return simple_qa(user_text)

# ─────────────────────── Streamlit UI ────────────────────────

def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_session_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict(role,str)]


def render_chat_tab() -> None:
    st.subheader("💬 Knowledge-Graph Chatbot")

    # 1) Geçmiş mesajları yazdır
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 2) Kullanıcı girdisi
    prompt = st.chat_input("Soru sorun…")
    if prompt is None:                        # Boşsa erkenden çık
        return

    # 3) Kullanıcı mesajını hemen göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4) Asistan balonu + spinner
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Yanıt hazırlanıyor…"):
            mode = st.radio(
                "Yanıt modu", ["Agent", "QA"],
                horizontal=True, index=0, key="chat_mode"
            )
            try:
                answer = generate_response(prompt,
                                            st.session_state.session_id,
                                            mode)
            except Exception as e:
                answer = f"🚨 Hata: {e}"

        placeholder.markdown(answer)

    # 5) Mesajı oturum hafızasına ekle
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


def render_risk_sentencing_workflow():

    st.title("NAJA: A Norm-Aware Artificial Intelligence Assistant for Judicial Risk Scoring and Sentencing Evaluation")
    def get_confidence_message(p, pred):
        if p >= 0.85:
            return f"{labels[pred]} (✔️ High Confidence)"
        elif p >= 0.6:
            return f"{labels[pred]} (⚠️ Moderate Confidence)"
        else:
            return f"{labels[pred]} (❗ Low Confidence – Other classes are also possible)"

    def plot_probabilities(proba):
        fig, ax = plt.subplots()
        bars = ax.bar(labels.values(), proba)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Risk Probability Distribution")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.0%}", ha='center', va='bottom')
        st.pyplot(fig)

    def calculate_base_sentence(qualifiers, mitigations, risk_pred, mot_change):

        base_sentence = "Life Imprisonment"
        if any(qualifiers.values()):
            base_sentence = "Aggravated Life Imprisonment"


        if (mitigations.get("self_defense") or mitigations.get("state_necessity") or
            (mitigations.get("minor_age") and mitigations.get("minor_age_group") == "Under 12") or
            (mitigations.get("deafness_muteness") and mitigations.get("deaf_age_group") == "Under 15") or
            mitigations.get("under_threat") or mitigations.get("under_drug") or
            (mitigations.get("mental_illness") and mitigations.get("mental_level") == "Fully")):
            base_sentence = "No Imprisonment"
            return base_sentence, base_sentence

        if mitigations.get("unjust_provocation"):
            ratio_map = {"Mild": Fraction(1,4), "Moderate": Fraction(1,2), "Severe": Fraction(3,4)}
            ratio = ratio_map.get(mitigations.get("provocation_level"), Fraction(0))
            if base_sentence == "Aggravated Life Imprisonment":
                if ratio == Fraction(1,4):
                    base_sentence = 24
                elif ratio == Fraction(2,4):
                    base_sentence = 21
                else :
                    base_sentence = 18
            elif base_sentence == "Life Imprisonment":
                if ratio == Fraction(1,4):
                    base_sentence = 18
                elif ratio == Fraction(2,4):
                    base_sentence = 15
                else :
                    base_sentence = 12
            else :
                base_sentence = base_sentence*(1-ratio)

        if ((mitigations.get("minor_age") and mitigations.get("minor_age_group") == "12-14") or
            (mitigations.get("deafness_muteness") and mitigations.get("deaf_age_group") == "15-17")):
            if base_sentence == "Aggravated Life Imprisonment":
                base_range = (12,15)
                base_sentence = risk_pred_mitigation_tuple(risk_pred, base_range)
            elif base_sentence == "Life Imprisonment":
                base_range = (9,11)
                base_sentence = risk_pred_mitigation_tuple(risk_pred, base_range)
            else :
                base_sentence = base_sentence/2
                if base_sentence > 7 :
                    base_sentence = 7
        elif ((mitigations.get("minor_age") and mitigations.get("minor_age_group") == "15-17") or
              (mitigations.get("deafness_muteness") and mitigations.get("deaf_age_group") == "18-21")):
            if base_sentence == "Aggravated Life Imprisonment":
                base_range = (18,24)
                base_sentence = risk_pred_mitigation_tuple(risk_pred, base_range)
            elif base_sentence == "Life Imprisonment":
                base_range = (12,15)
                base_sentence = risk_pred_mitigation_tuple(risk_pred, base_range)
            else :
                base_sentence = (base_sentence*2)/3
                if base_sentence > 12 :
                    base_sentence = 12
        if mitigations.get("mental_illness") and mitigations.get("mental_level") == "Partially":
            if base_sentence == "Aggravated Life Imprisonment":
                base_sentence = 25
            elif base_sentence == "Life Imprisonment":
                base_sentence = 20
            else :
                base_sentence = risk_pred_mitigation_int(risk_pred, base_sentence)

        if risk_pred == 0 :
            if base_sentence == "Aggravated Life Imprisonment":
                base_sentence = "Life Imprisonment"
            elif base_sentence == "Life Imprisonment":
                base_sentence = 25
            else :
                base_sentence = (base_sentence*5)/6
        elif risk_pred == 1 and mot_change:
            if base_sentence not in ["Aggravated Life Imprisonment", "Life Imprisonment"]:
                base_sentence = (base_sentence*7)/8
            elif base_sentence == "Aggravated Life Imprisonment":
                base_sentence = "Life Imprisonment"
            elif base_sentence == "Life Imprisonment":
                base_sentence = 25
        elif risk_pred == 1 and mot_change != True:
            if base_sentence not in ["Aggravated Life Imprisonment", "Life Imprisonment"]:
                base_sentence = (base_sentence*11)/12
        else :
            base_sentence = base_sentence

        if isinstance(base_sentence, (int, float)):
            return f"{base_sentence:.0f} years", base_sentence
        else:
            return base_sentence, base_sentence


    def risk_pred_mitigation_tuple(risk_pred, base_range):
        if risk_pred == 0:
            base_sentence = base_range[0] * 2/3
            return base_sentence
        elif risk_pred == 1:
            base_sentence = ((base_range[0] + base_range[1]) / 2) * 2/3
            return base_sentence
        else:
            base_sentence = base_range[1] * 2/3
            return base_sentence

    def risk_pred_mitigation_int(risk_pred,base_sentence):
        if risk_pred == 0:
            base_sentence = base_sentence * 5/6
            return base_sentence
        elif risk_pred == 1:
            base_sentence = base_sentence * 7/8
            return base_sentence
        else :
            base_sentence = base_sentence * 11/12
            return base_sentence

    def legal_norm_compliance(base_sentence):
        risk_pred = data["risk_pred"]
        qualifiers = data["qualifying_cases"]
        mitigations = data["mitigating_factors"]
        judge_type = data["judge_sentence_type"]
        judge_value = data["judge_sentence_value"]


        exculpatories = {
            "self_defense": mitigations.get("self_defense"),
            "state_necessity": mitigations.get("state_necessity"),
            "under_threat": mitigations.get("under_threat"),
            "under_drug": mitigations.get("under_drug"),
            "mental_illness_full": mitigations.get("mental_illness") and mitigations.get("mental_level") == "Fully",
            "minor_age_under12": mitigations.get("minor_age") and mitigations.get("minor_age_group") == "Under 12",
            "deaf_under15": mitigations.get("deafness_muteness") and mitigations.get("deaf_age_group") == "Under 15"
        }
        messages = []
        if any(exculpatories.values()) and judge_type != "No Imprisonment":
            messages.append(
                "\n⚖️ Exculpatory circumstances are present (e.g., self-defense, necessity, or diminished responsibility), "
                "yet a custodial sentence has been assigned. A re-evaluation of the judgment in light of Article 25 et seq. is recommended."
            )
        if any(qualifiers.values()) and judge_type!="Aggravated Life Imprisonment" and not any(mitigations.values()) :
            messages.append("\n⚠️ One or more qualifying circumstances exist (e.g., premeditation, public official victim), "
            "and no mitigating factors are present. According to TCK Article 82, the appropriate sentence is Aggravated Life Imprisonment."
            )
        if not any(qualifiers.values()) and judge_type=="Aggravated Life Imprisonment" :
            messages.append(
                "\n⚠️ No qualifying circumstance has been selected, yet the sentence assigned is Aggravated Life Imprisonment. "
                "Please verify compliance with the statutory aggravation conditions under TCK Article 82."
            )
        if (risk_pred in [1, 2]) and mitigations.get("discretionary_mitigation"):
            messages.append(
                "\n⚠️ Discretionary mitigation has been applied despite the individual being assessed as medium or high risk of recidivism. "
                "According to sentencing guidelines, this may not be appropriate and should be reconsidered."
            )
        if risk_pred == 0 and not mitigations.get("discretionary_mitigation") :
            messages.append(
                "\nℹ️ The defendant is assessed as low risk of recidivism. Consider applying discretionary mitigation "
                "to reflect positive rehabilitation potential."
            )
        if isinstance(judge_value, (int, float)) and base_sentence in ["Life Imprisonment","Aggravated Life Imprisonment"]:
            messages.append(
                "\n⚠️ The model suggests Life/Aggravated Life Imprisonment, yet the judge assigned a fixed-term sentence. "
                    "This is a potential severity mismatch and should be reviewed."
                )
        if isinstance(base_sentence,(int, float)) and judge_type != "Fixed":
            messages.append("\n⚠️ The model suggests fixed imprisonment, yet the judge assigned Life/Aggravated Life Imprisonment. This is a potential severity mismatch and should be reviewed.")
        if isinstance(judge_value, (int, float)) and isinstance(base_sentence, (int, float)):
            if float(judge_value * 2)/3 > float(base_sentence):
                messages.append(
                    "📈 The sentence imposed by the judge significantly exceeds the model’s recommended sentence range. "
                    "Consider reviewing the justification for this upward deviation."
                )
            if float(judge_value * 3)/2 < float(base_sentence):
                messages.append(
                    "\n📉 The judge’s sentence is substantially below the recommended range. "
                    "This may indicate under-sentencing; consider re-evaluating proportionality and deterrent effect."
                )

        return "\n".join(
            messages) if messages else "✅ The sentence appears to comply with both legal norms and the model’s risk and severity assessment."

    # --- Model ve Sabitler ---
    model = joblib.load("recidivism_xgb_pipeline.pkl")
    labels = {0: "🟢 Low Risk", 1: "🟡 Medium Risk", 2: "🔴 High Risk"}

    # --- Session State Başlatma ---
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "user_data" not in st.session_state:
        st.session_state.user_data = {}

    # === STEP 1: Recidivism Risk Prediction ===
    if st.session_state.step == 1:
        st.header("Step 1: Recidivism Risk Prediction Tool")
        st.markdown("""
            This tool estimates the likelihood of reoffending based on socio-demographic and behavioral data.  
            Please fill out the form and click the **Predict** button.
            """)
        col1, col2, col3 = st.columns(3)
        with col1:
            age_group = st.selectbox("Age Group", ["12-14","15-17","18-24","25-34","35-44","45-54","55-64","65+"])
            gender = st.selectbox("Gender", ["Male","Female"])
            race_ethnicity = st.selectbox("Ethnicity", ["Turk","Other"])
            # Dinamik seçimler…
            if age_group in ["12-14","15-17"]:
                education_options = ["Illiterate", "Literate without schooling", "Primary School", "Middle School"]
                marital_options = ["Single"]
                employment_options = ["Student","Employed"]
            elif age_group in ["18-24","25-34"]:
                education_options = ["Illiterate","…","Bachelor’s Degree"]
                marital_options = ["Single","Married","Divorced"]
                employment_options = ["Student","Employed","Unemployed"]
            else:
                education_options = ["Illiterate","…","Master/PhD"]
                marital_options = ["Single","Married","Divorced"]
                employment_options = ["Employed","Unemployed","Student","Retired"]
            education_level = st.selectbox("Education Level", education_options)
            marital_status = st.selectbox("Marital Status", marital_options)
        with col2:
            employment_status = st.selectbox("Employment Status", employment_options)
            housing_status = st.selectbox("Housing Status", ["Houseowner","Rent","Homeless"])
            prior_convictions = st.slider("Number of Prior Convictions", 0, 20, 0)
            juvenile_convictions = st.slider("Number of Juvenile Convictions", 0, 10, 0)
        with col3:
            no_prior = (prior_convictions == 0 and juvenile_convictions == 0)
            prior_probation_violation = st.checkbox("Probation Violation?", value=False, disabled=no_prior)
            prior_incarceration = st.checkbox("Has Been Incarcerated?", value=False, disabled=no_prior)
            if no_prior:
                st.caption("ℹ️ No prior convictions → violation/incarceration options are disabled.")
            is_child = (age_group == "12-14")
            has_dependents = st.checkbox("Has Dependents?", value=False, disabled=is_child)
            if is_child:
                st.caption("ℹ️ For Age < 15 depents are disabled.")
            substance_abuse_history = st.checkbox("Substance Abuse History?")
            mental_health_issues    = st.checkbox("Mental Health Issues?")
            gang_affiliation        = st.checkbox("Gang Affiliation?")
            aggression_history      = st.checkbox("Aggression History?")
            compliance_history      = st.checkbox("Compliant with Rules?")
            motivation_to_change    = st.checkbox("Motivated to Change?")
            stable_employment_past  = st.checkbox("Stable Employment History?")
            positive_social_support = st.checkbox("Positive Social Support?")
        if st.button("🧮 Predict"):
            df_input = pd.DataFrame([{
                "age_group": age_group, "gender": gender, "race_ethnicity": race_ethnicity,
                "education_level": education_level, "marital_status": marital_status,
                "employment_status": employment_status, "housing_status": housing_status,
                "has_dependents": has_dependents, "prior_convictions": prior_convictions,
                "juvenile_convictions": juvenile_convictions,
                "prior_probation_violation": prior_probation_violation,
                "prior_incarceration": prior_incarceration,
                "substance_abuse_history": substance_abuse_history,
                "mental_health_issues": mental_health_issues,
                "gang_affiliation": gang_affiliation, "aggression_history": aggression_history,
                "compliance_history": compliance_history,
                "motivation_to_change": motivation_to_change,
                "stable_employment_past": stable_employment_past,
                "positive_social_support": positive_social_support
            }])
            pred = model.predict(df_input)[0]
            proba = model.predict_proba(df_input)[0]
            st.markdown("---")
            st.subheader("🔍 Prediction Result")
            st.write(f"**Risk Level:** {get_confidence_message(proba[pred], pred)}")
            st.progress(int(proba[pred]*100))
            with st.expander("📊 Show Class Probabilities"):
                plot_probabilities(proba)
            st.session_state.user_data.update({
                "recidivism_input": df_input.to_dict(orient="records")[0],
                "risk_pred": int(pred),
                "risk_proba": proba.tolist(),
                "motivation_to_change": motivation_to_change
            })
        # Prediction yapılmadan Next butonunu kapatıyoruz
        next_disabled = "risk_pred" not in st.session_state.user_data
        if st.button("Next: Crime Type", disabled=next_disabled):
            st.session_state.step = 2
            st.rerun()

    # === STEP 2: Crime Type ===
    elif st.session_state.step == 2:
        st.header("Step 2: Crime Type")
        crime_type = st.selectbox("Select the committed crime", ["Intentional Killing (TCK 82)"])
        st.session_state.user_data["crime_type"] = crime_type

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Back"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Next: Qualifying Cases"):
                st.session_state.step = 3
                st.rerun()

    # === STEP 3: Qualifying Cases ===
    elif st.session_state.step == 3:
        st.header("Step 3: Qualifying Cases")

        qualifiers = {
            "premeditated_murder": st.checkbox("Premeditated killing", help="Planned in advance (TCK 82/a)"),
            "monstrous_feelings": st.checkbox("Killing with monstrous feelings or torture", help="Extreme cruelty (TCK 82/b)"),
            "killing_by_fire": st.checkbox("Killing by destructive means", help="e.g. bomb, flood, fire (TCK 82/c)"),
            "kin_murder": st.checkbox("Killing close relatives", help="Spouse, sibling, child, etc. (TCK 82/d)"),
            "child_murder": st.checkbox("Killing a child or defenseless person", help="Due to age or mental/physical incapacity"),
            "woman_murder": st.checkbox("Femicide", help="Due to gender (TCK 82/f)"),
            "public_duty_murder": st.checkbox("Victim was a public servant", help="Due to their public duty (TCK 82/g)"),
            "conceal_crime_murder": st.checkbox("To cover another crime", help="Or avoid detection (TCK 82/h)"),
            "failure_murder": st.checkbox("Rage after failed crime", help="Inflicted in emotional outburst (TCK 82/i)"),
            "blood_feud": st.checkbox("Blood feud motive", help="Revenge crimes due to clan conflicts (TCK 82/j)"),
            "tradition_murder": st.checkbox("Tradition/custom-based motive", help="Honor killing etc. (TCK 82/k)")
        }

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Back"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("Next: Mitigating or Exculpatory Factors"):
                st.session_state.user_data["qualifying_cases"] = qualifiers
                st.session_state.step = 4
                st.rerun()

    elif st.session_state.step == 4:
        st.header("Step 4: Mitigating or Exculpatory Factors")
        provocation_level = None
        minor_age_group   = None
        mental_level      = None
        deaf_age_group    = None

        self_defense         = st.checkbox("Legitimate Defence")
        state_necessity      = st.checkbox("State of Necessity")
        minor_age            = st.checkbox("Minor Age")
        if minor_age:
            minor_age_group = st.radio("Minor Age Group", ["Under 12","12-14","15-17"], horizontal=True)
        mental_illness       = st.checkbox("Mental Disorder")
        if mental_illness:
            mental_level = st.radio("Mental Disorder Level", ["Fully","Partially"])
        deafness_muteness    = st.checkbox("Deafness and Muteness")
        if deafness_muteness:
            deaf_age_group = st.radio("Deafness Age Group", ["Under 15","15-17","18-21"], horizontal=True)
        unjust_provocation    = st.checkbox("Unjust Provocation")
        if unjust_provocation:
            provocation_level = st.radio("Severity of provocation", ["Mild","Moderate","Severe"])
        error                 = st.checkbox("Mistake (Mistake of Fact or Law)")
        under_threat          = st.checkbox("Force and Violence, Menace and Threat")
        under_drug            = st.checkbox("Transitory Reasons and Being under Influence of Alcohol or Drugs")
        discretionary_mitigation = st.checkbox("Discretionary Mitigation")
        mitigation_value        = None
        if discretionary_mitigation:
            text = st.text_input("Mitigation Amount (fraction, max 1/6)", "0")
            try:
                mitigation_value = Fraction(text)
                if mitigation_value > Fraction(1,6):
                    st.warning("⚠️ Input can be max 1/6 according to TCK 62/1")
                else:
                    st.success(f"✅ Mitigation: {float(mitigation_value):.2%}")
            except:
                st.error("❌ Invalid fraction format.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Back"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("Next: Judge's Sentence Input"):
                st.session_state.user_data["mitigating_factors"] = {
                    "self_defense": self_defense,
                    "state_necessity": state_necessity,
                    "minor_age": minor_age,
                    "minor_age_group": minor_age_group,
                    "mental_illness": mental_illness,
                    "mental_level": mental_level,
                    "deafness_muteness": deafness_muteness,
                    "deaf_age_group": deaf_age_group,
                    "unjust_provocation": unjust_provocation,
                    "provocation_level": provocation_level,
                    "error": error,
                    "under_threat": under_threat,
                    "under_drug": under_drug,
                    "discretionary_mitigation": discretionary_mitigation,
                    "discretionary_mitigation_value": float(mitigation_value) if mitigation_value else None
                    }
                st.session_state.step = 5
                st.rerun()

    # === STEP 5: Judge's Sentence Input ===
    elif st.session_state.step == 5:
        st.header("Step 5: Judge's Sentence Input")
        sentence_type = st.selectbox("Select Sentence Type", ["Fixed Term (Years)","Life Imprisonment","Aggravated Life Imprisonment"])
        if sentence_type == "Fixed Term (Years)":
            years = st.number_input("Enter duration in years", min_value=1.0, step=0.5)
            st.session_state.user_data.update({
                "judge_sentence_type": "Fixed",
                "judge_sentence_value": years
            })
        else:
            st.session_state.user_data.update({
                "judge_sentence_type": sentence_type,
                "judge_sentence_value": None
            })
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅ Back"):
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("Next: Analysis"):
                st.session_state.step = 6
                st.rerun()

    # === STEP 6: Final Sentencing Analysis ===
    elif st.session_state.step == 6:
        st.header("Step 6: Final Sentencing Analysis")
        data = st.session_state.user_data
        # 1) Hakim’in Kararı
        st.subheader("⚖️ Judge's Proposed Sentence")
        if data["judge_sentence_type"] == "Fixed":
            st.write(f"Judge's proposed sentence: **{data['judge_sentence_value']} years**")
        else:
            st.write(f"Judge's proposed sentence: **{data['judge_sentence_type']}**")

        # 2) Model Önerisi
        st.subheader("📊 Suggested Sentence Analysis")
        suggested_string,suggested_value = calculate_base_sentence(
            data["qualifying_cases"],
            data["mitigating_factors"],
            data["risk_pred"],
            data["motivation_to_change"]
        )
        st.write(f"🧮 Suggested final sentence: **{suggested_string}**")

        # 3) Legal Uyumluluk
        st.subheader("📏 Legal Norm Compliance")
        st.write(legal_norm_compliance(suggested_value))

        st.subheader("🎯 Similar Suspects Graph")
        risk_class = data["risk_pred"]
        prob_pct = data["risk_proba"][risk_class] * 100.0
        from graph_helpers import similar_suspects_graph
        similar_suspects_graph(risk_class, prob_pct, k=3, tol_pct=5.0)

        # 4) Recidivism Risk Özeti
        st.subheader("🧠 Recidivism Risk")
        rp = data["risk_pred"]
        st.write(f"Predicted risk of reoffending: **{['Low','Medium','High'][rp]}** ({data['risk_proba'][rp]:.0%})")
        with st.expander("📊 Show Class Probabilities"):
            plot_probabilities(data["risk_proba"])

        # 5) Discretionary Mitigation Recommendation
        st.subheader("🧾 Discretionary Mitigation Recommendation")
        if rp == 0:
            st.success("🟢 Low risk: Discretionary mitigation may be applied.")
        elif rp == 1:
            st.warning("🟡 Medium risk: Discretionary mitigation may be reconsidered.")
        else:
            st.error("🔴 High risk: Discretionary mitigation should not be applied.")

        st.markdown("---")
        if st.button("🔁 Restart Application"):
            st.session_state.clear()
            st.rerun()


# ─────────────────────── main page layout ────────────────────────

def main():
    st.set_page_config(page_title="NAJA", layout="wide")

    init_session_state()

    tab_risk, tab_chat, tab_refs, tab_info = st.tabs([
        "Risk & Sentencing", "Chatbot", "Legal References", "Model Info"])

    with tab_risk:
        render_risk_sentencing_workflow()

    with tab_chat:
        render_chat_tab()

    # The other two tabs reuse the exact markdown/infos you already wrote.
    with tab_refs:
        from legal_references import render_legal_references  # optional helper
        render_legal_references()

    with tab_info:
        from model_info import render_model_info  # optional helper
        render_model_info()


if __name__ == "__main__":
    main()



