from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import Settings
import os

class DocumentProcessor:
    def __init__(self):
        self.settings = Settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP
        )
    
    def load_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        return self.text_splitter.split_documents(loader.load())
    
    def load_web(self, url):
        loader = WebBaseLoader(url)
        return self.text_splitter.split_documents(loader.load())
    
    def load_text(self, file_path):
        loader = TextLoader(file_path)
        return self.text_splitter.split_documents(loader.load())