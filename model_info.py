import streamlit as st

def render_model_info() -> None:
    st.subheader("🧠 Model Assumptions & Disclaimer")
    st.markdown("""
      - **Model**: Calibrated XGBoost classifier trained with socio-demographic & behavioral mock data 
      - **Classes**: 0 → Low, 1 → Medium, 2 → High Recidivism Risk  
       - **Pipeline**: Includes SMOTE, scaling, one-hot encoding  
      - **Disclaimer**: This tool is for educational/research purposes. It is not legal advice.  
       """)