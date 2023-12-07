import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoModel
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from tqdm import tqdm
import numpy as np
import random
import re
# import openai
import pathlib
from time import time, sleep
from typing import List
from xopen import xopen

def generate_response(model:AutoModel, tokenizer:AutoTokenizer, input:str, model_label:str, max_gen:int=100):
    with torch.no_grad():
        if model_label.endswith('base'):
            model_inputs = tokenizer(input, return_tensors="pt").to(device)
            output = model.generate(**model_inputs, max_new_tokens=max_gen)#generation_config=generation_config)
            response = tokenizer.decode(output[0], skip_special_tokens=True)[len(input):]
        else:
            if model_label.startswith('chatglm'):
                response, history = model.chat(tokenizer, input, history=[])
            else:
                model_inputs = tokenizer.apply_chat_template([{"role": "user", "content": input}], return_tensors="pt").to(device)
                output = model.generate(model_inputs, max_new_tokens=max_gen)#generation_config=generation_config)
                response = tokenizer.decode(output[0][model_inputs.shape[1]:], skip_special_tokens=True)
            
    return response

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
dataset2prompt = json.load(open("data/longbench/config/dataset2prompt_length.json", "r"))
dataset2maxlen = json.load(open("data/longbench/config/dataset2maxlen.json", "r"))

for model_name, model_label in [
    # Done
    # ("EleutherAI/gpt-neo-2.7B", "gpt-neo-2.7b-2k-base"), 
    # ("01-ai/Yi-6B", "yi-6b-4k-base"),
    
    # 10 document
    # ("THUDM/chatglm3-6b", "chatglm3-6b-8k-chat"),
    # ("THUDM/chatglm3-6b-base", "chatglm3-6b-32k-base"),
    # ("THUDM/chatglm3-6b-32k", "chatglm3-6b-32k-chat"),
    
    # ("mistralai/Mistral-7B-Instruct-v0.1", "mistral-7b-32k-inst"),
    
    # ("Salesforce/xgen-7b-4k-base", "xgen-7b-4k-base"),
    # ("Salesforce/xgen-7b-8k-base", "xgen-7b-8k-base"),
    # ("Salesforce/xgen-7b-8k-inst", "xgen-7b-8k-inst"),
    # # ("mosaicml/mpt-7b", "mpt-7b-2k-base"),
    # # ("mosaicml/mpt-7b-chat", "mpt-7b-2k-chat"),
    # # ("mosaicml/mpt-7b-instruct", "mpt-7b-2k-inst"),
    # ("tiiuae/falcon-7b", "falcon-7b-2k-base"),
    # ("tiiuae/falcon-7b-instruct", "falcon-7b-2k-inst"),
    # ("togethercomputer/RedPajama-INCITE-7B-Base", "redpajama-7b-2k-base"),
    # ("togethercomputer/RedPajama-INCITE-7B-Chat", "redpajama-7b-2k-chat"),
    # ("togethercomputer/RedPajama-INCITE-7B-Instruct", "redpajama-7b-2k-inst"),
    
    # # 20 document
    # # 4, 14
    # ("meta-llama/Llama-2-7b-hf", "llama-2-7b-4k-base"), 
    # ("meta-llama/Llama-2-7b-chat-hf", "llama-2-7b-4k-chat"), 
    ("togethercomputer/LLaMA-2-7B-32K", "llama-2-7b-32k-base"), 
    # ("togethercomputer/Llama-2-7B-32K-Instruct", "llama-2-7b-32k-inst"), 
    # ("lmsys/vicuna-7b-v1.5", "vicuna-7b-4k-chat"), 
    # ("lmsys/vicuna-7b-v1.5-16k", "vicuna-7b-16k-chat"), 
    
    # all
    # ("mosaicml/mpt-7b-8k", "mpt-7b-8k-base"),
    # ("mosaicml/mpt-7b-8k-chat", "mpt-7b-8k-chat"),
    # ("mosaicml/mpt-7b-8k-instruct", "mpt-7b-8k-inst"),
    # ("THUDM/chatglm2-6b", "chatglm2-6b-8k-chat"),
    # ("THUDM/chatglm2-6b-32k", "chatglm2-6b-32k-chat"),
    # ("01-ai/Yi-34B", "yi-34b-4k-base"),
    ("01-ai/Yi-6B-200K", "yi-6b-200k-base"),
    ("Qwen/Qwen-7B", "qwen-7b-8k-base"),
    ("mistralai/Mistral-7B-v0.1", "mistral-7b-32k-base"),
    ]:

    print('load model', model_label)
    # if model_label.startswith('llama-2-7b-32k') or model_label.startswith('mpt') or model_label.startswith('yi') or model_label.startswith('vicuna') or model_label.startswith('mistral'):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=torch.device(device), trust_remote_code=True, torch_dtype=torch.bfloat16)#, load_in_4bit=True)
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map=torch.device(device), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # max_length = int(model_label.split('-')[-2][:-1]) * 1024 - 100
    if model_label.startswith('yi'):
        max_length = 200 * 1024
    elif model_label.startswith('llama'):
        max_length = 32 * 1024
    else:
        max_length = 7 * 1024

    for dataset in [
        # "narrativeqa", 
                "qasper", 
                "multifieldqa_en", 
                # "multifieldqa_zh", 
                "hotpotqa", 
                "2wikimqa", 
                "musique",
                # "dureader", 
                # "qmsum", 
                # "multi_news", 
                # "gov_report", 
                # "vcsum", "trec", "triviaqa", #\ "samsum", 
                # "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
                ]:

        print(model_label, dataset)
        output_path = f"data/longbench/generation/{model_label}_{dataset}.jsonl.gz"
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(output_path):
            continue
        
        data = load_dataset('THUDM/LongBench', dataset, split='test')
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        with xopen(output_path, 'a', encoding="utf-8") as f_out:
            for obj_idx, json_obj in enumerate(tqdm(data)):
                
                prompt = prompt_format.format(**json_obj)
                if len(tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]) <= max_length:
                    pred = generate_response(model, tokenizer, prompt, model_label, max_gen)
                else:
                    pred = "SKIP"
                
                json.dump({"model_prompt": prompt, "model_documents": json_obj['context'], "model_answer": pred, "answers": json_obj["answers"], "length": json_obj["length"]}, f_out, ensure_ascii=False)
                f_out.write('\n')
        