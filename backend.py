from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

import asyncio
import nest_asyncio


nest_asyncio.apply()
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vector_store = None

def load_vector_store():
    global vector_store
    if os.path.exists("vector_db"):
        vector_store = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    return vector_store

def save_vector_store():
    global vector_store
    if vector_store:
        vector_store.save_local("vector_db")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str

def retrieval_node(state: ChatState):
    global vector_store
    messages = state['messages']
    last_message = messages[-1].content
    
    context = ""
    if vector_store:
        docs = vector_store.similarity_search(last_message, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
    
    return {"context": context}

def chat_node(state: ChatState):
    messages = state['messages']
    context = state.get('context', '')
    
    system_prompt = f"""You are a helpful assistant. Use the following context to answer questions when relevant:

Context:
{context}

If the context is relevant to the user's question, use it to provide a comprehensive answer. If not, respond normally based on your knowledge."""
    
    chat_messages = [HumanMessage(content=system_prompt)] + messages
    
    response = llm.invoke(chat_messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)
graph.add_node("retrieval", retrieval_node)
graph.add_node("chat", chat_node)

graph.add_edge(START, "retrieval")
graph.add_edge("retrieval", "chat")
graph.add_edge("chat", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def process_document(file_path):
    global vector_store
    
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    if vector_store is None:
        vector_store = FAISS.from_documents(splits, embeddings)
    else:
        vector_store.add_documents(splits)
    

    save_vector_store()

load_vector_store()