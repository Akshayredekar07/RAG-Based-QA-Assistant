from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import Settings
from pydantic import SecretStr

class ResponseGenerator:
    def __init__(self):
        self.settings = Settings()
        self.llm = ChatGroq(
            api_key=SecretStr(self.settings.GROQ_API_KEY) if self.settings.GROQ_API_KEY is not None else None,
            model=self.settings.GROQ_MODEL,
        )
        self.output_parser = StrOutputParser()
        
    def get_response_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        Answer based on this context:
        {context}
        
        Question: {question}
        
        Provide a detailed answer. If unsure, say you don't know.""")
        return prompt | self.llm | self.output_parser