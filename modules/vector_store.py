from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import Settings
from pydantic import SecretStr
import os

class VectorStoreManager:
    def __init__(self):
        self.settings = Settings()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretStr(self.settings.GEMINI_API_KEY) if self.settings.GEMINI_API_KEY is not None else None,
            task_type="retrieval_document"  
        )
    
    def create_vector_store(self, documents, store_name="default"):
        vector_store = FAISS.from_documents(documents, self.embeddings)
        self.save_vector_store(vector_store, store_name)
        return vector_store
    
    def save_vector_store(self, vector_store, store_name):
        save_path = os.path.join(self.settings.VECTOR_STORE_PATH, store_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vector_store.save_local(save_path)
    
    def load_vector_store(self, store_name="default"):
        load_path = os.path.join(self.settings.VECTOR_STORE_PATH, store_name)
        if os.path.exists(load_path):
            return FAISS.load_local(load_path, self.embeddings, allow_dangerous_deserialization=True)
        return None