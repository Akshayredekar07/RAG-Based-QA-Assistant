from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from config.settings import Settings

class CustomRetriever(BaseRetriever):
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.top_k = Settings().TOP_K_RETRIEVAL  # Fixed: Get from Settings
    
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.top_k)