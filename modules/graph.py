from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    documents: list[Document]
    response: str

class RAGWorkflow:
    def __init__(self, vector_store):
        from modules.retrieval import CustomRetriever
        from modules.generation import ResponseGenerator
        self.retriever = CustomRetriever(vector_store)
        self.generator = ResponseGenerator()
        self.workflow = self._build_workflow()
    
    def _retrieve_documents(self, state: GraphState) -> Dict:
        return {"documents": self.retriever.invoke(state["question"])}
    
    def _generate_response(self, state: GraphState) -> Dict:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        response = self.generator.get_response_chain().invoke({
            "context": context,
            "question": state["question"]
        })
        return {"response": response}
    
    def _build_workflow(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_response)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()
    
    def invoke(self, question: str):
        return self.workflow.invoke({
            "question": question,
            "documents": [],
            "response": ""
        })