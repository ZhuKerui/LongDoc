import os
import numpy as np
from collections import defaultdict, Counter
from time import time
import pprint
import json

GPT_MODEL_CHEAP = "gpt-4o-mini"
GPT_MODEL_EXPENSIVE = "gpt-4o"
DEFAULT_EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"
OPENAI_API_KEY_VARIABLE = "OPENAI_AUTO_SURVEY"
SPACY_MODEL = 'en_core_web_lg'
PARAGRAPH_SEP = '\n\n'
CHUNK_SEP = '\n\n\n\n'