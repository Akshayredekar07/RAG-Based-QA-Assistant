import streamlit as st
import uuid
import os
from langchain_core.messages import HumanMessage
from backend import retrieve_all_threads, process_document, chatbot

# Page config
st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖")

# Utility Functions
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Session State Initialization
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# Sidebar UI
st.sidebar.title("Gemini RAG Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"],
    help="Upload documents to add to the knowledge base"
)

if uploaded_file:
    with st.spinner("Processing document..."):
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            process_document(file_path)
            st.sidebar.success("✅ Document added to knowledge base!")
            # Clean up after processing
            os.remove(file_path)
        except Exception as e:
            st.sidebar.error(f"Error processing document: {str(e)}")

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    display_id = thread_id[:8] + "..."
    if st.sidebar.button(f"Chat {display_id}"):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = []
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangGraph &  Gemini")

# Main Chat Interface
st.title("Gemini RAG Chatbot")
st.markdown("Ask questions about your uploaded documents or anything else!")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Type your message here..."):
    st.session_state["message_history"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)


    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_placeholder = st.empty()

            full_response = ""
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="messages",
            ):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)


    st.session_state["message_history"].append({
        "role": "assistant",
        "content": full_response
    })
