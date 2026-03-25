import streamlit as st


def main():
    st.sidebar.title("Medical RAG")
    st.navigation(
        pages=[
            st.Page(
                page="pages/main_page.py",
                title="Main",
                icon="📇",
            ),
            st.Page(
                page="./pages/embed_page.py",
                title="Embed Document",
                icon="📄",
            ),
            st.Page(
                page="./pages/recommend_page.py",
                title="Recommend Treatment",
                icon="💊",
            ),
        ]
    ).run()


if __name__ == "__main__":
    main()
