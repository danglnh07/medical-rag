import tempfile
import os
import asyncio
import streamlit as st
from src.utils import inject_llm_env
from src.services import Embedder

# Inject environment variables
inject_llm_env(provider="gemini", api_key=st.secrets["API_KEY"])

st.set_page_config(page_title="Embed Document", page_icon="📄")

st.title("Embedding Document")

st.markdown(
    """
    You can embed your internal document here for the RAG system to digest
    and use for the recommendation system.

    > NOTE: our system currently support text document, all of your images
    > inside your documents will be ignored
    """
)

disease = st.selectbox(
    label="Disease",
    options=["CAP", "COPD", "AEB"],
)

file = st.file_uploader(
    label="Upload your document (prefer to be `.pdf` or `.docx`)",
    type=["pdf", "docx"],
    accept_multiple_files=False,
    help="Upload your document (prefer to be `.pdf` or `.docx`)",
)

if file is not None:
    if st.button(label="Submit"):
        with st.spinner("Embedding document..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(file.name)[1],
            ) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name

            try:
                embedder = Embedder(
                    qdrant_conn=st.secrets["QDRANT_URL"],
                    collection_name=st.secrets["COLLECTION_NAME"],
                    embedding_model=st.secrets["EMBEDDING_MODEL"],
                )

                success = asyncio.run(
                    embedder.embed_document(
                        disease=disease,
                        filepath=tmp_path,
                        filename=file.name,
                    )
                )

                if success:
                    st.success("Document embedded successfully!")
                else:
                    st.error("Failed to embed document. Check logs for details.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                pass
