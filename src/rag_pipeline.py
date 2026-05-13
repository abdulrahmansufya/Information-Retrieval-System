import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def load_pdf_documents(pdf_files) -> List[Document]:
    documents = []

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        file_name = getattr(pdf_file, "name", "uploaded_pdf")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_number,
                        },
                    )
                )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(documents)


def build_vector_store(chunks: List[Document]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def format_context(docs: List[Document]) -> str:
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")

        context_parts.append(
            f"[Source {i}: {source}, page {page}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def answer_question(vector_store, question: str) -> Dict[str, Any]:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    relevant_docs = retriever.invoke(question)
    context = format_context(relevant_docs)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful document question-answering assistant.

Answer the question using ONLY the context below.
If the answer is not in the context, say:
"I could not find the answer in the uploaded documents."

Rules:
- Be concise.
- Use clear language.
- Mention the source file and page when possible.

Context:
{context}

Question:
{question}
"""
    )

    llm = get_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    sources = []
    for doc in relevant_docs:
        sources.append(
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "preview": doc.page_content[:250],
            }
        )

    return {
        "answer": response.content,
        "sources": sources,
    }