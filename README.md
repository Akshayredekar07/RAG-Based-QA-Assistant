# RAG-Based QA Assistant

A modular Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **LangGraph**, **FAISS**, and **OpenAI**. This assistant answers user queries by retrieving relevant context from a custom document corpus and generating accurate, context-aware responses using a language model.

## 🔧 Features

- **RAG Pipeline**: Combines document retrieval and LLM-based response generation.
- **Modular Design**: Includes separate modules for document loading, vector storage, retrieval, generation, and graph flow.
- **Vector Store**: Uses FAISS for fast similarity search over embedded documents.
- **Graph Routing**: LangGraph is used to define an execution graph to manage the query pipeline efficiently.
- **LLM Integration**: Supports OpenAI GPT models for generation.


