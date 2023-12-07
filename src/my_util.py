import argparse
import os
import openai
from time import time, sleep

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "vicuna-v1.5-7b-16k", "chatgpt-16k", "chatgpt"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--sample', type=str, default=None, choices=['llm', 'dpr', 'llm-s', 'none', 'llm_sum'])
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    return parser.parse_args(args)

def output_path(args, dataset:str=None):
    out_path = 'pred_subsample_mqa'
    if args.e:
        out_path += '_e'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(f"{out_path}/{args.sample}"):
        os.makedirs(f"{out_path}/{args.sample}")
    if not os.path.exists(f"{out_path}/{args.sample}/{args.model}"):
        os.makedirs(f"{out_path}/{args.sample}/{args.model}")
    if dataset:
        out_path = f"{out_path}/{args.sample}/{args.model}/{dataset}.jsonl"
    else:
        out_path = f"{out_path}/{args.sample}/{args.model}/"
    return out_path


from openai import OpenAI



def openai_get(client:OpenAI, model_name, prompt, intermediate_sleep:float=1, final_sleep:float=2):
    start_time = time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    subsamples:str = response.choices[0].message.content
    token_cnt = response.usage.total_tokens
        
    while token_cnt / (time() - start_time) > (90000 / 60):
        sleep(intermediate_sleep)
        print('loop 1')
    if final_sleep:
        sleep(final_sleep)
    return subsamples