import streamlit as st
import os
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Free Document Chatbot", layout="wide")

st.title("📚 Free Web Document Chatbot (No API Keys Needed)")

uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, PPTX)",
    type=["pdf", "docx", "pptx"],
    accept_multiple_files=True
)

if st.button("Build Knowledge Base"):
    if not uploaded_files:
        st.error("Please upload at least one file.")
        st.stop()

    docs = []
    
    for file in uploaded_files:
        file_path = f"./temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            docs.extend(PyPDFLoader(file_path).load())
        elif file.name.endswith(".docx"):
            docs.extend(Docx2txtLoader(file_path).load())
        elif file.name.endswith(".pptx"):
            docs.extend(UnstructuredPowerPointLoader(file_path).load())

        os.remove(file_path)

    st.success(f"Loaded {len(docs)} pages/slides.")

    # Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Free model from HuggingFace
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.1, "max_length": 1024}
    )

    st.session_state.qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    st.success("Knowledge Base Ready!")

if "qa" in st.session_state:
    query = st.text_input("Ask a question from your documents:")
    if query:
        answer = st.session_state.qa.run(query)
        st.write("### Answer:")
        st.write(answer)
