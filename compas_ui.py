from fractions import Fraction
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Recidivism Risk Predictor", layout="wide")

tab1, tab2, tab3 = st.tabs(["Main", "Legal References", "Model Info"])

with tab2:
    st.subheader("📚 Legal References")
    st.markdown("""
    - **TCK 25**: Legitimate Defence and Necessity  
    - **TCK 28**: Force and Violence, Menace and Threat  
    - **TCK 29**: Unjust Provocation  
    - **TCK 30**: Mistake
    - **TCK 31**: Minors
    - **TCK 32**: Mental Disorder
    - **TCK 33**: Deafness and Muteness
    - **TCK 34**: Transitory Reasons and Being under Influence of Alcohol or Drugs
    - **TCK 61**: Determination of the Penalty
    - **TCK 81**: Intentional Killing
    - **TCK 81**: Qualified cases  
    """)
    st.info("""TCK Article 25: Legitimate Defence and Necessity ->
     (1)-No penalty shall be imposed upon an offender in respect of acts which were necessary to repel an unjust assault which is directed, carried out, certain to be carried out or to be repeated against a right to which he, or another, was entitled, provided such acts were proportionate to the assault, taking into account the situation and circumstances prevailing at the time. 
     (2)-No penalty shall be imposed upon an offender in respect of acts which were committed out of necessity, in order to protect against a serious and certain danger (which he has not knowingly caused) which was directed at a right to which he, or another, was entitled and where there was no other means of protection, provided that the means used were proportionate to the gravity and subject of the danger. """)

    st.info("""TCK Article 28: Force and Violence, Menace and Threat ->
     (1)-No penalty shall be imposed upon a person who commits a criminal offence as a result of intolerable or inevitable violence, or serious menace or gross threat. In such cases, the person involved in the use of force, violence, menace or threat shall be deemed to be the offender. """)
    st.info("""TCK Article 29: Unjust Provocation ->
    (1)-Any person who commits an offence in a state of anger or severe distress caused by an unjust act shall be sentenced to a penalty of imprisonment for a term of eighteen to twenty four years where the offence committed requires a penalty of aggravated life imprisonment and to a penalty of imprisonment for a term of twelve to eighteen years where the offence committed requires a penalty of life imprisonment. Otherwise the penalty to be imposed shall be reduced by one-quarter to three- quarters. 
    """)
    st.info("""TCK Article 30: Mistake ->
    (1)-Any person who, while conducting an act, is unaware of matters which constitute the actus reus of an offence, is not considered to have acted intentionally. Culpability with respect to recklessness shall be preserved in relation to such mistake. 
    (2)-Any person who is mistaken about matters which constitute an element of a qualified version of an offence, which requires an aggravated or mitigated sentence, shall benefit from such mistake.
    (3)-Any person who is inevitably mistaken about the conditions which, when satisfied, reduce or negate culpability shall benefit from such mistake.""")
    st.info("""TCK Article 31: Minors ->
    (1)-Minors under the age of twelve are exempt from criminal liability. While such minors cannot be prosecuted, security measures in respect of minors may be imposed. 
    (2)-(Amended on 29 June 2005 – By Article 5 of the Law no. 5377). Where a minor is older than twelve, but younger than fifteen, at the time of an offence, and he is either incapable of appreciating the legal meaning and consequences of his act or his capability to control his behavior is underdeveloped then he is shall be exempt from criminal liability. However, such minors may be subject to security measures specific to children. Where the minor has the capability to comprehend the legal meaning and result of the act and to control his behaviors in respective of his act, for offences requiring a penalty of aggravated life imprisonment, a term of twelve to fifteen years of imprisonment shall be imposed and for offences that require a penalty of life imprisonment, a term of nine to eleven years imprisonment shall be imposed. Otherwise the penalty to be imposed shall be reduced by half, save for the fact that for each act such penalty shall not exceed seven years.
    (3)-(Amended on 29 June 2005 – By Article 5 of the Law no. 5377). Where a minor is older than fifteen but younger than eighteen years at the time of the offence then for crimes that require a penalty of aggravated life imprisonment a term of eighteen to twenty four years of imprisonment shall be imposed and for offences that require a penalty of life imprisonment twelve to fifteen years of imprisonment shall be imposed. Otherwise the penalty to be imposed shall be reduced by one-third, save for the fact that the penalty for each act shall not exceed twelve years.
    """)
    st.info("""TCK Article 32: Mental Disorder ->
    (1)-A penalty shall not be imposed on a person who, due to mental disorder, cannot comprehend the legal meaning and consequences of the act he has committed, or if, in respect of such act, his ability to control his own behaviour was significantly diminished. However, security measures shall be imposed for such persons.
    (2)-Notwithstanding that it does not reach the extent defined in paragraph one, where a person’s ability to control his behaviour in respect of an act he has committed is diminished then a term of imprisonment for a term of twenty-five years where the offence committed requires a penalty of aggravated life imprisonment shall be imposed. Otherwise the penalty to be imposed may be reduced by no more than one-sixth. The penalty to be imposed may be enforced partially or completely as a security measure specific to mentally disordered persons, provided the length of the penalty remains the same.
    """)
    st.info("""TCK Article 33: Deafness and Muteness ->
    (1)-The provisions of this law which relate to minors under twelve years of age at the date of the offence shall also be applicable to deaf and mute persons under the age of fifteen. The provisions of this law which relate to minors who are over twelve years of age but under fifteen shall also be applicable to deaf and mute persons who are over fifteen years of age but under eighteen years of age. The provisions of this law which relate to minors over fifteen years of age but under eighteen of age shall be applied to deaf and mute persons who are over eighteen years of age but under twenty years of age.
    """)
    st.info("""TCK Article 34: Transitory Reasons and Being under Influence of Alcohol or Drugs
    (1)-Any person who is, because of a transitory reason or the effect of alcohol or drugs taken involuntarily, unable to comprehend the legal meaning and consequences of an act he has committed, or whose ability to control his behaviour regarding such act was significantly diminished, shall not be subject to a penalty.
    (2)-The provisions of the paragraph one shall not apply to a person who commits an offence under the effects of alcohol or drugs which have been taken voluntarily.""")
    st.info("""TCK Article 61: Determination of the Penalty ->
    (1)-In a particular case, the judge shall determine the basic penalty, between the minimum and maximum limits of the offence as defined by law, by considering the following factors:
        (a)-the manner in which the offence was committed;
        (b)-the means used to commit it;
        (c)-the time and place where the offence was committed;
        (d)-the importance and value of the subject of the offence;
        (e)-the gravity of the damage or danger;
        (f)-the degree of fault relating to the intent or recklessness;
        (g)-the object and motives of the offender.
    (4)-Where a qualified version of an offence creates more than one legal consequence which requires a penalty higher or lower than the basic version of that offence, the basic penalty is first increased then reduced.
    (5)-The penalty according to the above paragraphs will be finally determined by taking the following into consideration and in this order: attempt; jointly-committed offences; successive offences; unjust provocation; minor status; mental disorder, personal circumstances requiring a reduction of the penalty and discretionary mitigation.
    (7)-(Added on 29 June 2005 – By Article 7 of the Law no. 5377) The final penalty determined under this article for an offence that requires a specific of imprisonment, shall not exceed thirty years.
    (10)-Unless explicitly written in the law, penalties cannot be increased, decreased or converted.""")
    st.info("""TCK Article 81: Intentional Killing ->
    (1)-Any person who intentionally kills another shall be sentenced to life imprisonment.""")
    st.info("""TCK Article 82: Qualified cases ->
    (1)-If the act of intentional killing is committed: 
        (a)-With premeditation,
        (b)-Brutally or through torment;
        (c)-By causing fire, flood, destruction, sinking, bombing or by using nuclear, biological or chemical weapons;
        (d)-Against a direct ascendant, direct descendant, spouse or sibling;
        (e)-Against a child or against somebody who cannot protect himself physically or mentally;
        (f)-Against a pregnant woman, in knowledge of such pregnancy;
        (g)-Against a person because of the public service he performs;
        (h)-In order to conceal an offence, destroy evidence, facilitate the  commission of  another offence or prevent apprehension;
        (i)-(Added on 29 June 2005 – By Article 9 of the Law no. 5377) Out of  frustration for not being able to commit another offence;
        (j)-With the motive of a blood feud;
        (k)-With the motive of tradition
    the offender shall be sentenced to aggravated life imprisonment.""")

with tab3:
    st.subheader("🧠 Model Assumptions & Disclaimer")
    st.markdown("""
    - **Model**: Calibrated XGBoost classifier trained with socio-demographic & behavioral mock data 
    - **Classes**: 0 → Low, 1 → Medium, 2 → High Recidivism Risk  
    - **Pipeline**: Includes SMOTE, scaling, one-hot encoding  
    - **Disclaimer**: This tool is for educational/research purposes. It is not legal advice.  
    """)

with tab1:
    st.title("🧠 Recidivism Risk Prediction Tool")

    # --- Utility Fonksiyonlar ---
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
        # Adım 1: Ağırlaştırıcılar varsa doğrudan 'Aggravated Life'
        base_sentence = "Life Imprisonment"
        if any(qualifiers.values()):
            base_sentence = "Aggravated Life Imprisonment"

        # Adım 2: Tam indirim durumları
        if (mitigations.get("self_defense") or mitigations.get("state_necessity") or
            (mitigations.get("minor_age") and mitigations.get("minor_age_group") == "Under 12") or
            (mitigations.get("deafness_muteness") and mitigations.get("deaf_age_group") == "Under 15") or
            mitigations.get("under_threat") or mitigations.get("under_drug") or
            (mitigations.get("mental_illness") and mitigations.get("mental_level") == "Fully")):
            base_sentence = "No Imprisonment"
            return base_sentence, base_sentence

        # Adım 3: Haksız tahrik indirimi
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
        # Adım 4: Küçük yaş / işitme-engel indirimi aralığı
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
        #Adım 5: Takdiri indirim
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

        # Exculpatory check
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
                "⚖️ Exculpatory circumstances are present (e.g., self-defense, necessity, or diminished responsibility), "
                "yet a custodial sentence has been assigned. A re-evaluation of the judgment in light of Article 25 et seq. is recommended."
            )
        if any(qualifiers.values()) and judge_type!="Aggravated Life Imprisonment" and not any(mitigations.values()) :
            messages.append("⚠️ One or more qualifying circumstances exist (e.g., premeditation, public official victim), "
            "and no mitigating factors are present. According to TCK Article 82, the appropriate sentence is Aggravated Life Imprisonment."
            )
        if not any(qualifiers.values()) and judge_type=="Aggravated Life Imprisonment" :
            messages.append(
                "⚠️ No qualifying circumstance has been selected, yet the sentence assigned is Aggravated Life Imprisonment. "
                "Please verify compliance with the statutory aggravation conditions under TCK Article 82."
            )
        if (risk_pred in [1, 2]) and mitigations.get("discretionary_mitigation"):
            messages.append(
                "⚠️ Discretionary mitigation has been applied despite the individual being assessed as medium or high risk of recidivism. "
                "According to sentencing guidelines, this may not be appropriate and should be reconsidered."
            )
        if risk_pred == 0 and not mitigations.get("discretionary_mitigation") :
            messages.append(
                "ℹ️ The defendant is assessed as low risk of recidivism. Consider applying discretionary mitigation "
                "to reflect positive rehabilitation potential."
            )
        if isinstance(judge_value, (int, float)) and base_sentence in ["Life Imprisonment","Aggravated Life Imprisonment"]:
            messages.append(
                "⚠️ The model suggests Life/Aggravated Life Imprisonment, yet the judge assigned a fixed-term sentence. "
                    "This is a potential severity mismatch and should be reviewed."
                )
        if isinstance(base_sentence,(int, float)) and judge_type != "Fixed":
            messages.append("⚠️ The model suggests fixed imprisonment, yet the judge assigned Life/Aggravated Life Imprisonment. This is a potential severity mismatch and should be reviewed.")
        if isinstance(judge_value, (int, float)) and isinstance(base_sentence, (int, float)):
            if float(judge_value * 2)/3 > float(base_sentence):
                messages.append(
                    "📈 The sentence imposed by the judge significantly exceeds the model’s recommended sentence range. "
                    "Consider reviewing the justification for this upward deviation."
                )
            if float(judge_value * 3)/2 < float(base_sentence):
                messages.append(
                    "📉 The judge’s sentence is substantially below the recommended range. "
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


        # 4) Recidivism Risk Özeti
        st.subheader("🧠 Recidivism Risk")
        rp = data["risk_pred"]
        st.write(f"Predicted risk of reoffending: **{['Low','Medium','High'][rp]}** ({data['risk_proba'][rp]:.0%})")

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