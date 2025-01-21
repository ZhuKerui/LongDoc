from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyMuPDFLoader

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import langchain_core.documents
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import tracing_v2_enabled

from langgraph.graph import END, StateGraph, START

from pydantic import BaseModel, Field
import os
import evaluate
from openai import OpenAI
import numpy as np
from collections import defaultdict, Counter
from time import time

GPT_MODEL_CHEAP = "gpt-4o-mini"
GPT_MODEL_EXPENSIVE = "gpt-4o"
DEFAULT_EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Load the ROUGE metric
def eval_rouge(predictions:list[str], references:list[list[str]]):
    return evaluate.load('rouge').compute(predictions=predictions, references=references)

# Dataset Sample
class Sample(BaseModel):
    doc_file: str = ''
    doc_str: str = ''
    questions: list[str] = []
    answers: list[str] = []

# LangGraph Pipeline
class MyPipeline:
    def __init__(self, doc_file:str=None, doc_str:str=None, enable_trace:bool=False, project_name:str=None):
        self.enable_trace = enable_trace
        self.project_name = project_name
        self.doc_file = doc_file
        self.doc_str = doc_str
        # Load LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.environ.get("OPENAI_AUTO_SURVEY", default=None))
        self.load_langgraph()
        
    def update_doc(self, doc_file:str=None, doc_str:str=None):
        raise NotImplementedError
        
    def __call__(self, question:str):
        if self.enable_trace and self.project_name is not None:
            with tracing_v2_enabled(self.project_name):
                return self.invoke(question)
        else:
            return self.invoke(question)
    
    def invoke(self, question:str):
        pass
    
    def clear_up(self):
        pass
    
    def load_langgraph(self):
        pass
    
    @staticmethod
    def remove_tab(text:str):
        return text.replace('    ', '').strip()