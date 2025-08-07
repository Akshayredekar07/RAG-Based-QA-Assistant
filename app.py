import streamlit as st
from modules.document_loader import DocumentProcessor
from modules.vector_store import VectorStoreManager
from modules.graph import RAGWorkflow
import os

# Initialize
document_processor = DocumentProcessor()
vector_store_manager = VectorStoreManager()
os.makedirs(vector_store_manager.settings.VECTOR_STORE_PATH, exist_ok=True)

# UI
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📚 RAG Knowledge Assistant (Ollama + Groq)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Setup")
    doc_type = st.radio("Document Type:", ["PDF", "Web URL", "Text"])
    
    if doc_type == "PDF":
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file and st.button("Process PDF"):
            with st.spinner("Processing..."):
                temp_path = f"temp_{pdf_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                docs = document_processor.load_pdf(temp_path)
                vector_store_manager.create_vector_store(docs, "knowledge_base")
                st.success("PDF processed!")
                os.remove(temp_path)
    
    elif doc_type == "Web URL":
        url = st.text_input("Enter URL:")
        if url and st.button("Process URL"):
            with st.spinner("Processing..."):
                docs = document_processor.load_web(url)
                vector_store_manager.create_vector_store(docs, "knowledge_base")
                st.success("Web content processed!")
    
    elif doc_type == "Text":
        text_file = st.file_uploader("Upload Text", type=["txt"])
        if text_file and st.button("Process Text"):
            with st.spinner("Processing..."):
                temp_path = f"temp_{text_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(text_file.getbuffer())
                docs = document_processor.load_text(temp_path)
                vector_store_manager.create_vector_store(docs, "knowledge_base")
                st.success("Text processed!")
                os.remove(temp_path)

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload documents to begin!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about your documents:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    vector_store = vector_store_manager.load_vector_store("knowledge_base")
    if vector_store:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    rag = RAGWorkflow(vector_store)
                    result = rag.invoke(prompt)
                    response = result["response"]
                except Exception as e:
                    response = f"Error: {str(e)}"
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload documents first!")