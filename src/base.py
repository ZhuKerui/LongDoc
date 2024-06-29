from typing import List, Tuple, Dict, Callable, Set, Literal, Any
import networkx as nx
from tqdm import tqdm
import numpy as np
import itertools
from collections import defaultdict, Counter
import json
import torch
from pathlib import Path
import os
import string, re
from time import time
import statistics
import random
import pandas as pd
import matplotlib.pyplot as plt

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_text_splitters import SpacyTextSplitter
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain import hub

# from rank_bm25 import BM25Okapi
from nltk import word_tokenize, sent_tokenize
from datasets import load_dataset
from nltk.corpus import wordnet

from transformers import AutoTokenizer

os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["TOKENIZERS_PARALLELISM"] = 'false'

DEFAULT_LLM = "mistralai/Mistral-7B-Instruct-v0.3"