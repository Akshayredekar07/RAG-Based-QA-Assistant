import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Groq settings (for chat)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = "mixtral-8x7b-32768"
    
    # Gemini settings (for embeddings)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "models/embedding-001"
    
    # Vector store settings
    VECTOR_STORE_PATH = "vector_store"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 3

settings = Settings()