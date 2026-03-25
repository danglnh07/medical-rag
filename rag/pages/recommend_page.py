import asyncio
import streamlit as st
from src.services import Recommender

st.set_page_config(page_title="Recommend Treatment", page_icon="💊")

st.title("Recommend Treatment Protocol")

st.markdown(
    """
    Input the patient's disease and clinical picture to receive treatment
    protocol and medicine recommendations.
    """
)

disease = st.selectbox(
    label="Disease",
    options=["CAP", "COPD", "AEB"],
)

clinical_picture = st.text_area(
    label="Clinical Picture",
    placeholder=(
        "Enter clinical information such as severity, risk factors, "
        "CFR, antibiotic allergy history, etc."
    ),
    height=300,
)

provider = st.selectbox(
    label="LLM Provider",
    options=["OpenAI", "Anthropic", "Gemini"],
)

model = st.text_input(label="LLM Model", value="gemini/gemini-2.5-flash")
api_key = st.text_input(label="LLM API Key", type="password")

if st.button(label="Get Recommendation", disabled=not clinical_picture.strip()):
    with st.spinner("Analyzing and generating recommendation..."):
        try:
            recommender = Recommender(
                qdrant_conn=st.secrets["QDRANT_URL"],
                collection_name=st.secrets["COLLECTION_NAME"],
                embedding_model=st.secrets["EMBEDDING_MODEL"],
            )

            result = asyncio.run(
                recommender.recommend_medicine(
                    disease=disease,
                    clinical_picture=clinical_picture,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                )
            )

            if result is None:
                st.error("No recommendation found")
            else:
                st.header("Treatment Protocol")
                st.write(result.recommended_documents[0])

                st.subheader("Empiric Antibiotic")
                st.write(result.empiric_antibiotic)

                st.subheader("Treatment Site")
                st.write(result.treatment_site if result.treatment_site else "N/A")

                st.subheader("Medicine Information")
                med = result.medicine_result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dosage", f"{med.dosage} {med.unit}")
                with col2:
                    st.metric("Dosing Interval", f"Every {med.dosing_interval} hours")

                st.write(f"**Route of Administration:** {med.route_of_administration}")

                if result.recommended_documents:
                    st.subheader("Reference Documents")
                    for doc in result.recommended_documents:
                        st.write(f"- {doc}")
        except Exception as e:
            st.error(f"Error: {e}")
