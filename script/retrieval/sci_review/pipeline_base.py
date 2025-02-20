from .paper_tools import *

import os
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import END, StateGraph, START

# LangGraph Pipeline
class MyPipeline:
    def __init__(self, enable_trace:bool=False, project_name:str=None, llm_model:str=GPT_MODEL_CHEAP):
        self.enable_trace = enable_trace
        self.project_name = project_name
        self.doc_manager:DocManager = None
        # Load LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=os.environ.get(OPENAI_API_KEY_VARIABLE, default=None))
        # self.load_langgraph()
        
    def __call__(self, question:str):
        if self.enable_trace and self.project_name is not None:
            with tracing_v2_enabled(self.project_name):
                return self.invoke(question)
        else:
            return self.invoke(question)
    
    def invoke(self, question:str):
        pass
    
    def load_langgraph(self):
        pass
    
    @staticmethod
    def remove_tab(text:str):
        return text.replace('    ', '').strip()