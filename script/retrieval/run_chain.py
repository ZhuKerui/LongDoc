import random
import json
from openai import OpenAI
import os
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.my_util import openai_get

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
# client = OpenAI(api_key='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')

prompt_template = '''The text below describes the order of numbers in several chains. Find the complete chain that starts with number {start_number}. Only output the chain and seperate the numbers with space.

{description}

The chain starts with {start_number} is: '''

test_file = 'data/chain_generation/fixed_chain/test_left2right_400_20.txt'

with open(test_file) as f_in:
    data = [json.loads(l) for l in f_in]
    
for data_idx in tqdm(range(len(data))):
    if data_idx < 0:
        continue
    if data_idx >= 400:
        break
    question = data[data_idx]['question']
    chains = data[data_idx]['chains']
    prompt = prompt_template.format(start_number=question, description='\n'.join(data[data_idx]['list']))
    response = openai_get(client, 'gpt-3.5-turbo', prompt)
    
    with open(test_file.replace('test', 'prediction'), 'a') as f_out:
        f_out.write(json.dumps({
            'chains': chains,
            'question': question,
            'prediction': response.strip()
        }) + '\n')