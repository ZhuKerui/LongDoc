cur_folder = './'

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from huggingface_hub import login
login('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')
# login(os.environ['HUGGINGFACE_ACCESS_TOKEN'])
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig

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

T = TypeVar("T")


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
    question: str, documents: List[Document], file_name: str = None
):
    with open(file_name) as f:
        prompt_template = f.read().rstrip("\n")

    # Format the documents into strings
    formatted_documents = []
    for document_index, document in enumerate(documents):
        formatted_documents.append(f"Document [{document_index+1}](Title: {document.title}) {document.text}")
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

for model_name, model_label in [
    # ("lmsys/vicuna-7b-v1.5-16k", "vicuna-7b-16k"), 
    # ("lmsys/vicuna-7b-v1.5", "vicuna-7b"), 
    ("meta-llama/Llama-2-7b-hf", "llama-2-7b"), 
    # ("meta-llama/Llama-2-7b-chat-hf", "llama2-7b-chat"), 
    # ("togethercomputer/LLaMA-2-7B-32K", "llama-2-7b-32k"), 
    # ("togethercomputer/Llama-2-7B-32K-Instruct", "llama-2-7b-32k-instruct")
    ]:


    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    total_doc_num = 10
    prompt_file = 'closedbook_qa.prompt'
    start_idx = 0
    end_idx = 100
    last_break_idx = 0
    generation_config = GenerationConfig(max_new_tokens=20)
    prompt_path = cur_folder + f'data/lost-in-the-middle/prompts/{prompt_file}'
    
    for position in [0]:

        print(model_label, position)
        # Create directory for output path if it doesn't exist.
        input_path = cur_folder + 'data/lost-in-the-middle/qa_data/%d_total_documents/nq-open-%d_total_documents_gold_at_%d.jsonl.gz' % (total_doc_num, total_doc_num, position)
        output_path = cur_folder + 'data/lost-in-the-middle/qa_predictions/%s-prediction-%d-%d-%d-%d-%s.jsonl.gz' % (model_label, total_doc_num, position, start_idx, end_idx, prompt_file)
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
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

            prompt = get_qa_prompt(
                question,
                documents,
                file_name=prompt_path,
            )

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

        with torch.no_grad():
            with xopen(output_path, "a") as f:
                for example, model_documents, prompt in tqdm(zip(examples, all_model_documents, prompts), total=len(prompts)):

                    model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                    output = model.generate(**model_inputs, generation_config=generation_config)
                    response = tokenizer.decode(output[0], skip_special_tokens=True)

                    output_example = deepcopy(example)
                    # Add some extra metadata to the output example
                    output_example["model_prompt"] = prompt
                    output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
                    output_example["model_answer"] = response[len(prompt):]
                    f.write(json.dumps(output_example) + "\n")
