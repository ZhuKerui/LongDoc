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
random.seed(0)

T = TypeVar("T")

irrelevant_operation = 'random'
total_doc_num = 30
prompt_file = 'qa_remind.prompt'
remind_interval = 3
start_idx = 0
end_idx = 100
last_break_idx = 0
device = 'cuda:1'

@dataclass(frozen=True)
class Document:
    title: str
    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    hasanswer: Optional[bool] = None
    isgold: Optional[bool] = None
    original_retrieval_index: Optional[int] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        data = deepcopy(data)
        if not data:
            raise ValueError("Must provide data for creation of Document from dict.")
        id = data.pop("id", None)
        score = data.pop("score", None)
        # Convert score to float if it's provided.
        if score is not None:
            score = float(score)
        return cls(**dict(data, id=id, score=score))


def get_qa_prompt(
    question: str, documents: List[Document], file_name: str, irrelevant_operation:str = '', remind_interval:int=1
):
    with open(file_name) as f:
        prompt_template = f.read().rstrip("\n")

    if irrelevant_operation:
        idx:int
        document:Document
        for idx, document in enumerate(documents):
            if document.isgold:
                break
        irrelevant_documents = documents[:idx] + documents[idx+1:]
        if irrelevant_operation == 'reverse':
            irrelevant_documents.reverse()
        elif irrelevant_operation == 'random':
            random.shuffle(irrelevant_documents)
        irrelevant_documents.insert(idx, document)
        documents = irrelevant_documents
        
    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
    if file_name == 'qa_remind.prompt':
        formatted_documents_with_remind = []
        for did, document in enumerate(formatted_documents):
            if did % remind_interval == 0:
                formatted_documents_with_remind.append(f'Reminder of question: {question}')
            formatted_documents_with_remind.append(document)
        formatted_documents = formatted_documents_with_remind
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

METRICS = [
    (best_subspan_em, "best_subspan_em"),
]

def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)

def generate_response(model:AutoModel, tokenizer:AutoTokenizer, input:str, model_label:str, generation_config:GenerationConfig=None):
    with torch.no_grad():
        if model_label.endswith('base'):
            model_inputs = tokenizer(input, return_tensors="pt").to(device)
            output = model.generate(**model_inputs, max_new_tokens=100)#generation_config=generation_config)
            response = tokenizer.decode(output[0], skip_special_tokens=True)[len(input):]
        else:
            if model_label.startswith('chatglm'):
                response, history = model.chat(tokenizer, input, history=[])
            else:
                model_inputs = tokenizer.apply_chat_template([{"role": "user", "content": input}], return_tensors="pt").to(device)
                output = model.generate(model_inputs, max_new_tokens=100)#generation_config=generation_config)
                response = tokenizer.decode(output[0][model_inputs.shape[1]:], skip_special_tokens=True)
            
    return response

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
    ("Salesforce/xgen-7b-8k-base", "xgen-7b-8k-base"),
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
    # ("meta-llama/Llama-2-7b-hf", "llama2-7b-4k-base"), 
    # ("meta-llama/Llama-2-7b-chat-hf", "llama2-7b-4k-chat"), 
    ("togethercomputer/LLaMA-2-7B-32K", "llama2-7b-32k-base"), 
    # ("togethercomputer/Llama-2-7B-32K-Instruct", "llama2-7b-32k-inst"), 
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
    if model_label.startswith('llama2-7b-32k') or model_label.startswith('mpt') or model_label.startswith('yi') or model_label.startswith('vicuna'):
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=torch.device(device), trust_remote_code=True, torch_dtype=torch.bfloat16)#, load_in_4bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=torch.device(device), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # generation_config = GenerationConfig(max_new_tokens=20)
    prompt_path = cur_folder + f'data/lost-in-the-middle/prompts/{prompt_file}'
    
    for position in [0, 4, 9, 14, 19, 24, 29]:

        print(model_label, position)
        # Create directory for output path if it doesn't exist.
        input_path = cur_folder + f"data/lost-in-the-middle/qa_data/{total_doc_num}_total_documents/nq-open-{total_doc_num}_total_documents_gold_at_{position}.jsonl.gz"
        irrelevant_label = ''
        if irrelevant_operation:
            if irrelevant_operation == 'random':
                irrelevant_label = '-ra'
            elif irrelevant_operation == 'reverse':
                irrelevant_label = '-re'
        output_path = cur_folder + f"data/lost-in-the-middle/qa_predictions/{model_label}-prediction-{total_doc_num}-{position}-{start_idx}-{end_idx}-{prompt_file if prompt_file != 'qa_remind.prompt' else 'qa_remind' + str(remind_interval) + '.prompt'}{irrelevant_label}.jsonl.gz"
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(output_path):
            continue
        
        # Fetch all of the prompts
        with xopen(input_path) as fin:
            samples = []
            for idx, line in enumerate(fin):
                if idx >= start_idx:
                    if (idx - start_idx) < last_break_idx:
                        continue
                    if idx >= end_idx:
                        break
                else:
                    continue
                input_example = json.loads(line)
                samples.append(input_example)

        examples = []
        prompts = []
        all_model_documents = []
        for input_example in samples:
            question = input_example["question"]

            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))

            if prompt_file != 'qa_remind.prompt':
                prompt = get_qa_prompt(
                    question,
                    documents,
                    file_name=prompt_path,
                    irrelevant_operation=irrelevant_operation
                )
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    file_name=prompt_path,
                    irrelevant_operation=irrelevant_operation,
                    remind_interval=remind_interval
                )

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

        with torch.no_grad():
            with xopen(output_path, "a") as f:
                for example, model_documents, prompt in tqdm(zip(examples, all_model_documents, prompts), total=len(prompts)):
                    response = generate_response(model, tokenizer, prompt, model_label)
                    output_example = deepcopy(example)
                    # Add some extra metadata to the output example
                    output_example["model_prompt"] = prompt
                    output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
                    output_example["model_answer"] = response
                    f.write(json.dumps(output_example) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()