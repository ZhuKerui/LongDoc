cur_folder = './'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from huggingface_hub import login
login('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')
# login(os.environ['HUGGINGFACE_ACCESS_TOKEN'])
from transformers import AutoTokenizer, GenerationConfig, AutoModel, AutoModelForCausalLM

import dataclasses
import json
import logging
import math
import pathlib
import random
import sys
from copy import deepcopy
import torch
from tqdm import tqdm
from xopen import xopen
from pydantic.dataclasses import dataclass
from typing import List, Optional, Tuple, Type, TypeVar
import string
import statistics
import regex
import gc
from src.my_util import openai_get, OpenAI

client = OpenAI()

with open('data/lost-in-the-middle/prompts/eval.prompt') as f_in:
    prompt_format = f_in.read()

if not os.path.exists('data/lost-in-the-middle/qa_predictions/llm_eval/'):
    os.mkdir('data/lost-in-the-middle/qa_predictions/llm_eval/')

temp = [
    'falcon-7b-2k-base-prediction-10-4-0-100-qa.prompt.jsonl.gz',
    'falcon-7b-2k-base-prediction-10-9-0-100-qa.prompt.jsonl.gz',
    'gpt-neo-2.7b-2k-base-prediction-10-0-0-100-qa.prompt.jsonl.gz',
    'gpt-neo-2.7b-2k-base-prediction-10-9-0-100-qa.prompt.jsonl.gz',
]
eval_files = [f for f in os.listdir('data/lost-in-the-middle/qa_predictions/') if 'base' in f and f in temp]
eval_files.sort()
for eval_file in eval_files:
    print(eval_file)
    output_file = f'data/lost-in-the-middle/qa_predictions/llm_eval/{eval_file}'
    temp_output_file = f'data/lost-in-the-middle/qa_predictions/llm_eval/temp_{eval_file}'
    if os.path.exists(output_file):
        continue
    start_idx = 0
    if os.path.exists(temp_output_file):
        with open(temp_output_file) as f_in:
            start_idx = len(f_in.readlines())
        
    all_examples = []
    with xopen(f'data/lost-in-the-middle/qa_predictions/{eval_file}') as f_in:
        for line in f_in:
            input_example = json.loads(line)
            all_examples.append(input_example)
            
    with open(temp_output_file, 'a') as f_out:
        for e_idx, example in enumerate(tqdm(all_examples)):
            if e_idx < start_idx:
                continue
            prediction = example['model_answer'].split('\n')[0].strip()
            if prediction:
                prompt = prompt_format.format(question=example['question'], answers=str(example['answers']), prediction=prediction)
                response = openai_get(client, 'gpt-3.5-turbo', prompt, final_sleep=1)
                if response.lower().strip().startswith('yes'):
                    score = 1
                elif response.lower().strip().startswith('no'):
                    score = 0
                else:
                    score = -1
                f_out.write(json.dumps({'prompt': prompt, 'score': score, 'response': response}) + '\n')
            else:
                f_out.write(json.dumps({'prompt': '', 'score': 0, 'response': ''}) + '\n')
    
    os.rename(temp_output_file, output_file)
