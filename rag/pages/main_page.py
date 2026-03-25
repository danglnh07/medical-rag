import streamlit as st

st.set_page_config(page_title="Home", page_icon="📇")

st.markdown(""" 
# RAG medical system

This a RAG built for a hospital `AMS` system, which help doctors in
diagnosing patients and recommending treatment protocols from a private
knowledge base.

## Basic features

### Embed internal document

Doctor can embed their internal documents (treatment protocols) into the
system for the recommendation system to use.

Basic flow: document -> chunking -> embed chunks -> add to vector store

### Recommend treatment protocol and medicine based on clinical picture

Doctor can input information about client:

- Disease (e.g. `CAP`, `COPD`, `AEB`,...)
- Clinical picture (e.g. severity, Risk factors for antibiotic-resistant infections,
`eCFR`, Antibiotic allergy history,...)

The system will return:

- Empirical treatment protocol
- Alternative treatment protocol (if any)
- Medicine information:
  - Dosage
  - Route of administration
  - Dosing interval

---

> Quick access:
>
> 1. Qdrant dashboard: http://localhost:6333/dashboard

""")
