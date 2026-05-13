import streamlit as st

from src.rag_pipeline import (
    load_pdf_documents,
    split_documents,
    build_vector_store,
    answer_question,
)

st.set_page_config(
    page_title="Smart Document Retrieval System",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Smart Document Retrieval System")

st.caption("Chat with your PDF documents using AI")

# =========================
# Session State
# =========================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed" not in st.session_state:
    st.session_state.processed = False

# =========================
# Sidebar
# =========================

with st.sidebar:

    st.header("📄 Upload Documents")

    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_button = st.button("Submit & Process")

    if process_button:

        if not pdf_files:
            st.warning("Please upload at least one PDF file.")

        else:

            with st.spinner("Reading PDF files..."):
                documents = load_pdf_documents(pdf_files)

            if not documents:
                st.error("No readable text found in the uploaded PDFs.")

            else:

                with st.spinner("Splitting documents into chunks..."):
                    chunks = split_documents(documents)

                with st.spinner("Building FAISS vector index..."):
                    st.session_state.vector_store = build_vector_store(chunks)

                st.session_state.processed = True

                st.success(
                    f"Done. Processed {len(documents)} pages into {len(chunks)} chunks."
                )

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

# =========================
# Main Page
# =========================

if not st.session_state.processed:
    st.info(
        "Upload PDF files from the sidebar, then click Submit & Process."
    )

# =========================
# Display Chat History
# =========================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            with st.expander("Sources used"):

                for i, source in enumerate(message["sources"], start=1):

                    st.write(f"Source {i}")
                    st.write(f"File: {source['source']}")
                    st.write(f"Page: {source['page']}")

                    st.code(source["preview"])

# =========================
# User Input
# =========================

user_question = st.chat_input(
    "Ask a question about your uploaded documents..."
)

if user_question:

    if st.session_state.vector_store is None:

        st.warning("Please upload and process PDF files first.")

    else:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.spinner(
            "Retrieving relevant chunks and generating answer..."
        ):

            result = answer_question(
                vector_store=st.session_state.vector_store,
                question=user_question,
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )

        with st.chat_message("assistant"):

            st.write(result["answer"])

            with st.expander("Sources used"):

                for i, source in enumerate(
                    result["sources"],
                    start=1,
                ):

                    st.write(f"Source {i}")
                    st.write(f"File: {source['source']}")
                    st.write(f"Page: {source['page']}")

                    st.code(source["preview"])