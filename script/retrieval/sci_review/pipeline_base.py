from .paper_tools import *
from .data_base import Sample

import os
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import END, StateGraph, START

# LangGraph Pipeline
class MyPipeline:
    # Nodes
    AGENT = "agent"
    # State
    MESSAGES = "messages"
    
    def __init__(self, enable_trace:bool=False, project_name:str=None, llm_model:str=GPT_MODEL_CHEAP):
        self.enable_trace = enable_trace
        self.project_name = project_name
        # Load LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=0, api_key=os.environ.get(OPENAI_API_KEY_VARIABLE, default=None))
        # self.load_langgraph()
        
    def __call__(self, question:str):
        if self.enable_trace and self.project_name is not None:
            with tracing_v2_enabled(self.project_name):
                return self.invoke(question)
        else:
            return self.invoke(question)
    
    def invoke(self, question:str, chunks:list[str]):
        pass
    
    def load_langgraph(self, **kwargs):
        pass
    
    @staticmethod
    def remove_tab(text:str):
        return text.replace('    ', '').strip()
    
    @staticmethod
    def dump_process(process:list[dict], dump_file:str):
        pass
            
    @staticmethod
    def load_process(dump_file:str) -> list[dict]:
        pass
    
    @staticmethod
    def get_chunk_ids_from_process(
        process:list[dict],
        relevant_only:bool=True
    ) -> list[int]:
        pass
            