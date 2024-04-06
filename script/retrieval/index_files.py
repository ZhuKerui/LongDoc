from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Any, Union
import torch
from nltk import sent_tokenize, word_tokenize
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI
import networkx as nx
from tqdm import tqdm
import numpy as np
import itertools
from collections import defaultdict, Counter
from rouge_metric import PyRouge
import re, json, string, copy, os
from rank_bm25 import BM25Okapi
from prompt import *


def cal_rouge(hp, ref):
    return PyRouge().evaluate([hp], [[ref]])

def read_jsonline(file:str):
    with open(file) as f_in:
        return [json.loads(l) for l in f_in]

def read_json(file:str):
    with open(file) as f_in:
        return json.load(f_in)
    
def write_json(file, obj):
    with open(file, 'w') as f_out:
        json.dump(obj, f_out)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class DocIndex:
    def __init__(self, graph:nx.DiGraph, paragraphs:List[str], summary:List[str], paragraph_embs:np.ndarray, pid2nodes:List[List[str]]) -> None:
        self.graph = graph
        self.paragraphs = paragraphs
        self.summary = summary
        self.paragraph_embs = paragraph_embs
        self.pid2nodes = pid2nodes
        self.bm25 = BM25Okapi([[w.lower() for w in word_tokenize(p)] for p in paragraphs])
    
    
# class DocSplit:
#     def split_trec(text:str):
#         lines = text.splitlines()
#         return ['\n'.join(lines[i * 2 : i * 2 + 1]) for i in range(len(lines) // 2)]

#     def split_triviaqa(text:str):
#         lines = text.splitlines()
#         paragraphs = []
#         paragraph = []
#         lid = 0
#         while lid < len(lines):
#             paragraph.append(lines[lid])
#             if lines[lid] == 'Answer:':
#                 lid += 1
#                 paragraph.append(lines[lid])
#                 paragraphs.append('\n'.join(paragraph))
#                 paragraph.clear()
#             lid += 1
#         return paragraphs

#     def split_samsum(text:str):
#         paragraphs = []
#         paragraph = []
#         for line in text.splitlines():
#             paragraph.append(line)
#             if line.startswith('Summary: '):
#                 paragraphs.append('\n'.join(paragraph))
#                 paragraph.clear()
#         return paragraphs

#     paragraph_sep_map = {
#         'qasper': '\n', 
#         'multifieldqa_zh': '\n', 
#         'qmsum': '\n', 
#         'multi_news': '\n', 
#         'vcsum': '\n', 
#         'trec': (split_trec, '\n'), 
#         'triviaqa': (split_triviaqa, '\n'), 
#         'samsum': (split_samsum, '\n'), 
#     }
    
#     def __init__(self, llm_name:str) -> None:
#         self.llm_name = llm_name
#         self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        
#     def _append_paragraph(self, paragraphs:list, tokenized_p:List[str]):
#         paragraph = self.llm_tokenizer.decode(tokenized_p)
#         paragraphs.append(paragraph)
#         tokenized_p.clear()
        
#     def get_task_paragraph_sep(self, task_name:str):
#         sep = self.paragraph_sep_map.get(task_name, '\n\n')
#         if not isinstance(sep, str):
#             func, sep = sep
#         return sep
    
#     def split_context_to_paragraphs(self, context:str, task_name:str):
#         sep = self.paragraph_sep_map.get(task_name, '\n\n')
#         if isinstance(sep, str):
#             return context.split(sep)
#         else:
#             func, sep = self.paragraph_sep_map[task_name]
#             return func(context)
        
#     def split_single_paragraph(self, text:str, paragraph_size:int=300, is_natural_language:bool=True):
#         splited_paragraphs:List[str] = []
#         splited_paragraph = []
#         sentences:List[str] = sent_tokenize(text) if is_natural_language else text.split('\n')
#         for sent in sentences:
#             tokenized_s = self.llm_tokenizer.encode(sent)[1:]
#             if len(tokenized_s) <= paragraph_size:
#                 if len(splited_paragraph) + len(tokenized_s) > paragraph_size:
#                     self._append_paragraph(splited_paragraphs, splited_paragraph)
#                 splited_paragraph.extend(tokenized_s)
#             else:
#                 if splited_paragraph:
#                     self._append_paragraph(splited_paragraphs, splited_paragraph)
#                 chunk_size = (len(tokenized_s) - 1) // paragraph_size + 1
#                 for i in range(chunk_size - 1):
#                     self._append_paragraph(splited_paragraphs, tokenized_s[i * paragraph_size: (i+1) * paragraph_size])
#                 splited_paragraph = tokenized_s[(chunk_size - 1) * paragraph_size:]
#         return splited_paragraphs, splited_paragraph
        
#     def split_paragraphs(self, text:str, task_name:str, paragraph_size:int=300):
#         reformated_paragraphs:List[str] = []
#         completion_labels:List[bool] = []
#         reformated_paragraph = []
        
#         paragraph_sep = self.get_task_paragraph_sep(task_name)
#         paragraphs = text.split(paragraph_sep)
#         for p in paragraphs:
#             tokenized_p = self.llm_tokenizer.encode(p + paragraph_sep)[1:]
#             if len(tokenized_p) <= paragraph_size:
#                 if len(reformated_paragraph) + len(tokenized_p) > paragraph_size:
#                     self._append_paragraph(reformated_paragraphs, reformated_paragraph)
#                     completion_labels.append(True)
#                 reformated_paragraph.extend(tokenized_p)
#             else:
#                 if reformated_paragraph:
#                     self._append_paragraph(reformated_paragraphs, reformated_paragraph)
#                     completion_labels.append(True)
#                 splited_paragraphs, splited_paragraph = self.split_single_paragraph(p, paragraph_size)
#                 reformated_paragraphs.extend(splited_paragraphs)
#                 completion_labels.extend([False] * len(splited_paragraphs))
#                 reformated_paragraph = splited_paragraph
                
#         if reformated_paragraph:
#             self._append_paragraph(reformated_paragraphs, reformated_paragraph)
#             completion_labels.append(True)
#         return reformated_paragraphs, completion_labels
    

class LLMServer:
    def __init__(self, llm_name:str="mistralai/Mistral-7B-Instruct-v0.2") -> None:
        self.llm_name = llm_name
        if 'gpt' in self.llm_name:
            # self.doc_split = DocSplit("mistralai/Mistral-7B-Instruct-v0.2")
            self.llm_server = OpenAI()
        else:
            # self.doc_split = DocSplit(self.llm_name)
            
            # Set OpenAI's API key and API base to use vLLM's API server.
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8000/v1"
            self.llm_server = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
    
    def __call__(self, prompt:str, n:int=1, temperature:float=0.0, return_str:bool=True):
        chat_response = self.llm_server.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            n=n,
            temperature=temperature,
        )
        return chat_response.choices[0].message.content if return_str else chat_response

    
class Dataset:
    def __init__(self, dataset_name:str, llm_name:str="mistralai/Mistral-7B-Instruct-v0.2", split:str='train') -> None:
        self.dataset_name = dataset_name
        self.answer_format = None
        self.question_type = None
        self.data = []
        self.llm_server = LLMServer(llm_name)
        self.split = split
        self.load_split()
        self.data_dir = Path(os.path.join(self.dataset_name, self.split))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_split(self):
        raise NotImplementedError
    
    @staticmethod
    def parse_pause_point(text:str):
        text = text.strip("Break point: ")
        if text[0] != '<':
            return None
        for i, c in enumerate(text):
            if c == '>':
                if text[1:i].isnumeric():
                    return int(text[1:i])
                else:
                    return None
        return None
    
    def paragraph_parser(self, article: str) -> List[str]:
        """Parse Gutenberg articles."""
        lines = []
        previous_line = None
        for i, line in enumerate(article.split('\n')):
            line = line.strip()
            original_line = line
            if line == '':
                if previous_line == '':
                    line = '\n'
                else:
                    previous_line = original_line
                    continue
            previous_line = original_line
            lines.append(line)
        return (' '.join(lines)).split('\n')
    
    @staticmethod
    def count_words(text:str):
        """Simple word counting."""
        return len(text.split())
    
    
    def get_questions_and_answers(self, sample:dict) -> Tuple[List[str], List[str]]:
        raise NotImplementedError
    
    def get_article(self, sample:dict) -> str:
        raise NotImplementedError


    def pagination(
        self,
        article:str,
        word_limit=600,
        start_threshold=280,
        verbose=True,
        allow_fallback_to_last=True
    ):
        paragraphs = self.paragraph_parser(article)

        i = 0
        pages = []
        while i < len(paragraphs):
            preceding = "" if i == 0 else "...\n" + '\n'.join(pages[-1])
            passage = [paragraphs[i]]
            wcount = self.count_words(paragraphs[i])
            j = i + 1
            while wcount < word_limit and j < len(paragraphs):
                wcount += self.count_words(paragraphs[j])
                if wcount >= start_threshold:
                    passage.append(f"<{j}>")
                passage.append(paragraphs[j])
                j += 1
            passage.append(f"<{j}>")
            end_tag = "" if j == len(paragraphs) else paragraphs[j] + "\n..."

            pause_point = None
            if wcount < 350:
                pause_point = len(paragraphs)
            else:
                prompt = prompt_pagination_template.format(preceding, '\n'.join(passage), end_tag)
                response = self.llm_server(prompt=prompt).strip()
                pause_point = self.parse_pause_point(response)
                if pause_point and (pause_point <= i or pause_point > j):
                    print(f"prompt:\n{prompt},\nresponse:\n{response}\n")
                    print(f"i:{i} j:{j} pause_point:{pause_point}")
                    pause_point = None
                if pause_point is None:
                    if allow_fallback_to_last:
                        pause_point = j
                    else:
                        raise ValueError(f"prompt:\n{prompt},\nresponse:\n{response}\n")

            page = paragraphs[i:pause_point]
            pages.append(page)
            if verbose:
                print(f"Paragraph {i}-{pause_point-1}", page)
            i = pause_point
        if verbose:
            print(f"[Pagination] Done with {len(pages)} pages")
        return pages
    
    def eval(self, gen:str, answer:str):
        raise NotImplementedError
    
    def load_and_eval_result(self, task_start:int, task_end:int, prefix:dict={}):
        results = defaultdict(list)
        for r_tool in ['index', 'dpr', 'gist']:
            for task_i in range(task_start, task_end):
                generation_file = os.path.join(self.data_dir, f'{prefix.get(r_tool, "")}generation_{r_tool}_s_{task_i}.jsonl')
                if os.path.exists(generation_file):
                    _, answers = self.get_questions_and_answers(self.data[task_i])
                    temp_results = []
                    temp_dict = {}
                    steps = []
                    for line in read_jsonline(generation_file):
                        if line[0] == 'query':
                            temp_dict[line[0]] = line[1]
                        elif line[0] == 'generation':
                            temp_dict['steps'] = steps
                            temp_dict[line[0]] = line[1]
                            temp_dict['q_i'] = len(temp_results)
                            temp_dict['gold'] = answers[temp_dict['q_i']]
                            temp_dict['task_i'] = task_i
                            pred, score = self.eval(line[1], temp_dict['gold'])
                            temp_dict['predict'] = pred
                            temp_dict['acc'] = score
                            temp_results.append(temp_dict)
                            temp_dict = {}
                            steps = []
                        else:
                            steps.append(line)
                    results[r_tool].extend(temp_results)
            if r_tool in results:
                print(r_tool, sum([r['acc'] for r in results[r_tool]]) * 1. / len(results[r_tool]))
        return results
    

class QualityDataset(Dataset):
    
    # Fields that are straight text copies from raw example to processed example.
    _ONE2ONE_FIELDS = (
        'article',
        'article_id',
        'set_unique_id',
        'writer_id',
        'source',
        'title',
        'topic',
        'url',
        'writer_id',
        'author',
    )
    
    bracketed_lowercase_letters_set = set(
        [f"({l})" for l in string.ascii_lowercase]
    )  # {"(a)", ...}
    bracketed_uppercase_letters_set = set(
        [f"({l.upper()})" for l in string.ascii_lowercase]
    )  # {"(A)", ...}

    choices = ['(A)', '(B)', '(C)', '(D)']
    
    def __init__(self, llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'train') -> None:
        super().__init__('quality', llm_name, split)
        self.answer_format = '''If (A) is correct, answer with \"Answer: (A) ...\"\nIf (B) is correct, answer with \"Answer: (B) ...\"\nIf (C) is correct, answer with \"Answer: (C) ...\"\nIf (D) is correct, answer with \"Answer: (D) ...\"'''
        self.question_type = 'multiple choice question'
        
    def load_split(self):
        self.data = []
        with open(f'../../data/QuALITY/QuALITY.v1.0.1.htmlstripped.{self.split}', 'r') as f:
            for line in f.readlines():
                j = json.loads(line)
                fields = {k: j[k] for k in self._ONE2ONE_FIELDS}
                fields.update({
                    'questions': [q['question'] for q in j['questions']],
                    'question_ids': [q['question_unique_id'] for q in j['questions']],
                    'difficults': [q['difficult'] for q in j['questions']],
                    'options': [q['options'] for q in j['questions']],
                })

                fields.update({
                    'gold_labels': [q['gold_label'] for q in j['questions']],
                    'writer_labels': [q['writer_label'] for q in j['questions']],
                })

                self.data.append(fields)
                
    def get_index_from_symbol(self, answer):
        """Get the index from the letter symbols A, B, C, D, to extract answer texts.

        Args:
            answer (str): the string of answer like "(B)".

        Returns:
            index (int): how far the given choice is from "a", like 1 for answer "(B)".
        """
        answer = str(answer).lower()
        # extract the choice letter from within bracket
        if answer in self.bracketed_lowercase_letters_set:
            answer = re.findall(r"\(.*?\)", answer)[0][1]
        index = ord(answer) - ord("a")
        return index

    def get_questions_and_answers(self, sample: dict):
        return ['\n'.join([question] + [f"{ol} {o}" for ol, o in zip(self.choices, option)]) for option, question in zip(sample['options'], sample['questions'])], sample['gold_labels']
        
    def get_article(self, sample: dict) -> str:
        return sample['article']
    
    def eval(self, gen: str, answer: str):
        gen = gen.strip()
        if 'answer: ' in gen.lower():
            gen = gen.lower().split('answer: ', 1)[-1].split()[0].strip('().,:').upper()
        elif 'answer is' in gen.lower():
            gen = gen.lower().split('answer is', 1)[-1].split()[0].strip('().,:').upper()
        
        if gen in 'ABCD':
            predict = 'ABCD'.index(gen) + 1
        else:
            predict = 0
            # print(gen)
        return predict, predict == answer
    

class NarrativeQADataset(Dataset):
    
    def __init__(self, llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'test') -> None:
        super().__init__('narrativeqa', llm_name, split)
        self.answer_format = '''Generate your answer and explanation using the following format:\n"Answer: your answer to the question ...\nExplanation: your explanation or reasoning ...". Your answer should be brief and specific.'''
        self.question_type = 'question'
        
    def load_split(self):
        self.data = list(load_dataset('THUDM/LongBench', 'narrativeqa', split='test'))
        
    def paragraph_parser(self, article: str) -> List[str]:
        return article.split('\n\n')
        
    def get_questions_and_answers(self, sample: dict) -> Tuple[List[str], List[str]]:
        return [sample['input']], [sample['answers']]
    
    def get_article(self, sample: dict) -> str:
        return sample['context']
    
    def eval(self, gen: str, answers: List[str]):
        gen = gen.strip().lower()
        f1 = 0.
        pred = ''
        if gen.startswith('answer: '):
            gen = gen.split('answer: ', 1)[-1]
            gen = gen.split('explanation:')[0].strip()
            pred = gen
            gen = normalize_answer(gen).split()
            
            for answer in answers:
                answer = normalize_answer(answer).split()
                common = Counter(gen) & Counter(answer)
                num_same = sum(common.values())
                if num_same == 0:
                    temp_f1 = 0
                else:
                    precision = 1.0 * num_same / len(gen)
                    recall = 1.0 * num_same / len(answer)
                    temp_f1 = (2 * precision * recall) / (precision + recall)
                if temp_f1 > f1:
                    f1 = temp_f1
        return pred, f1
    
class MuSiQueDataset(Dataset):
    
    def __init__(self, llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'test') -> None:
        super().__init__('musique', llm_name, split)
        self.answer_format = '''Generate your answer and explanation using the following format:\n"Answer: your answer to the question ...\nExplanation: your explanation or reasoning ...". Your answer should be brief and specific.'''
        self.question_type = 'question'
        
    def load_split(self):
        self.data = list(load_dataset('THUDM/LongBench', 'musique', split='test'))
        
    def paragraph_parser(self, article: str) -> List[str]:
        return article.split('\n\n')
        
    def get_questions_and_answers(self, sample: dict) -> Tuple[List[str], List[str]]:
        return [sample['input']], [sample['answers']]
    
    def get_article(self, sample: dict) -> str:
        return sample['context']
    
    def eval(self, gen: str, answers: List[str]):
        gen = gen.strip().lower()
        f1 = 0.
        pred = ''
        if gen.startswith('answer: '):
            gen = gen.split('answer: ', 1)[-1]
            gen = gen.split('explanation:')[0].strip()
            pred = gen
            gen = normalize_answer(gen).split()
            
            for answer in answers:
                answer = normalize_answer(answer).split()
                common = Counter(gen) & Counter(answer)
                num_same = sum(common.values())
                if num_same == 0:
                    temp_f1 = 0
                else:
                    precision = 1.0 * num_same / len(gen)
                    recall = 1.0 * num_same / len(answer)
                    temp_f1 = (2 * precision * recall) / (precision + recall)
                if temp_f1 > f1:
                    f1 = temp_f1
        return pred, f1
    
        
class LooGlEDataset(Dataset):
    def __init__(self, llm_name: str = "mistralai/Mistral-7B-Instruct-v0.2", split: str = 'test') -> None:
        super().__init__('loogle', llm_name, split)
        self.answer_format = '''Generate your answer and explanation using the following format:\n"Answer: your answer to the question ...\nExplanation: your explanation or reasoning ...". Your answer should be brief and specific.'''
        self.question_type = 'question'
        
    def load_split(self):
        self.data = [self.parse_qa_pairs(data) for data in load_dataset('bigainlco/LooGLE', 'longdep_qa', split='test')]

    def parse_qa_pairs(self, data:dict):
        qa_pairs:str = data['qa_pairs']
        pairs = []
        for qa_pair in qa_pairs.strip('[{}]').split('}, {'):
            A = qa_pair.index("'A': ")
            Q = qa_pair.index("'Q': ")
            S = qa_pair.index("'S': ")
            T = qa_pair.index("'type': ")
            qs = qa_pair[Q + 5: A].strip(' ,"\'').replace("\\n", '\n')
            a_s = qa_pair[A + 5:T].strip(' ,"\'').replace("\\n", '\n')
            ts = qa_pair[T + 8:S].strip(' ,"\'').replace("\\n", '\n')
            ss = [s_.replace("\\n", '\n').replace('\\', '') for s_ in re.split('", "|", \'|\', "|\', \'', qa_pair[S + 5:].strip('[]\'"'))]
            pairs.append({'Q': qs, 'A': a_s, 'type': ts, 'S': ss})
        return {'input': data['input'], 'title': data['title'], 'qa_pairs': pairs}
    
    def paragraph_parser(self, article: str) -> List[str]:
        return sent_tokenize(article) if '\n' not in article else article.split('\n')
    
    def get_questions_and_answers(self, sample: dict) -> Tuple[List[str]]:
        return [qa_pair['Q'] for qa_pair in sample['qa_pairs']], [qa_pair['A'] for qa_pair in sample['qa_pairs']]
    
    def get_article(self, sample: dict) -> str:
        return sample['input']


def log_info(log_file:str, tag:str, info:Any):
    if log_file is not None:
        with open(log_file, 'a') as f_out:
            f_out.write(json.dumps([tag, info]))
            f_out.write('\n')
                
class ReadingAgent:
    def __init__(self, dataset:Dataset, llm_name:str="mistralai/Mistral-7B-Instruct-v0.2", model_type:str = 'gpt') -> None:
        self.llm_server = LLMServer(llm_name)
        self.model_type = model_type
        self.dataset = dataset
    

    def gisting(self, example, pages, verbose=True):
        article = self.dataset.get_article(example)
        word_count = Dataset.count_words(article)

        shortened_pages = []
        for i, page in enumerate(pages):
            prompt = prompt_shorten_template.format('\n'.join(page))
            response = self.llm_server(prompt)
            shortened_text = response.strip()
            shortened_pages.append(shortened_text)
            if verbose:
                print("[gist] page {}:".format(i), shortened_text, flush=True)
        shortened_article = '\n'.join(shortened_pages)
        gist_word_count = Dataset.count_words(shortened_article)
        if verbose:
            print("Shortened article:\n", shortened_article, flush=True)
        output:dict = copy.deepcopy(example)
        output.update({'word_count': word_count, 'gist_word_count': gist_word_count, 'shortened_pages': shortened_pages, 'pages': pages})
        if verbose:
            print(f"compression rate {round(100.0 - gist_word_count/word_count*100, 2)}% ({gist_word_count}/{word_count})")
        return output
    
    
    def main(self, query:str, page_file:str, gist_file:str, log_file:str=None, lookup_method:str='s'):
        pages = read_json(page_file)
        shortened_pages = read_json(gist_file)['shortened_pages']
        expanded_shortened_pages = shortened_pages[:]
        
        log_info(log_file, 'query', query)
        
        if lookup_method == 'p':
            menu = '\n'.join([f"<Page {i}>\n{shortened_text}" for i, shortened_text in enumerate(shortened_pages)])
            log_info(log_file, 'menu', menu)

            prompt_lookup = prompt_parallel_lookup_template.format(menu, query)
            # log_info(log_file, 'lookup_prompt', prompt_lookup)

            page_ids = []
            response = self.llm_server(prompt=prompt_lookup).strip()
            log_info(log_file, 'retrieval_command', response)
            
            if 'I want to look up Page'.lower() in response.lower():
                start = response.lower().index('I want to look up Page'.lower()) + len('I want to look up Page')
                page_ids_str = response[start:].split('to', 1)[0].split()
                page_ids = []
                for p in page_ids_str:
                    if p.strip(',.[]').isnumeric():
                        page_id = int(p.strip(',.[]'))
                        if page_id >= 0 and page_id < len(pages):
                            page_ids.append(page_id)

            # Memory expansion after look-up, replacing the target shortened page with the original page
            if len(page_ids) > 0:
                for page_id in page_ids:
                    expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])
        else:
            retrieved_passage_nums = []
            for _ in range(7):
                menu = '\n'.join([f"<Page {i}>\n{shortened_text}" for i, shortened_text in enumerate(expanded_shortened_pages)])
                log_info(log_file, 'menu', menu)
                
                prompt_lookup = prompt_sequential_lookup_template.format(menu, query, retrieved_passage_nums)
                # log_info(log_file, 'lookup_prompt', prompt_lookup)
                
                response = self.llm_server(prompt=prompt_lookup).strip()
                log_info(log_file, 'retrieval_command', response)
                
                page_id:List[str] = re.findall(r'page \d+', response.lower())
                
                if 'STOP' in response or not page_id:
                    log_info(log_file, 're-read complete', 'yes')
                    break
                elif page_id:
                    # Memory expansion after look-up, replacing the target shortened page with the original page
                    page_id = int(page_id[0].split()[1])
                    if page_id >= 0 and page_id < len(shortened_pages) and page_id not in retrieved_passage_nums:
                        if len(retrieved_passage_nums) == 6:
                            log_info(log_file, 're-read complete', 'no')
                        else:
                            expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])
                            retrieved_passage_nums.append(page_id)
                            log_info(log_file, 'retrieved_pids', retrieved_passage_nums)

        expanded_shortened_article = '\n'.join(expanded_shortened_pages)
        log_info(log_file, 'retrieval_result', expanded_shortened_article)

        response = self.llm_server(prompt=prompt_answer_template.format(question_type=self.dataset.question_type, article=expanded_shortened_article, question=query, answer_format=self.dataset.answer_format))
        response = response.strip()
        log_info(log_file, 'generation', response)
    
    
class LongDoc:
    
    def __init__(self, dataset:Dataset, retriever_model_name:str='facebook/contriever', llm_name:str="mistralai/Mistral-7B-Instruct-v0.2", device='cpu') -> None:
        self.dataset = dataset
        self.llm_server = LLMServer(llm_name)
        # self.nlp = spacy.load('en_core_web_lg')#, disable=['attribute_ruler', 'lemmatizer', 'ner'])
        # self.nlp.add_pipe('coreferee')
        # self.nlp = spacy.load("en_core_web_lg")
        # self.nlp.add_pipe("fastcoref", 
        #      config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda:1', 'enable_progress_bar': False}
        # )
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_model_name)
        self.retriever_model = AutoModel.from_pretrained(retriever_model_name)
        if device == 'cpu':
            self.device = None
        else:
            self.device = torch.device(device)
            self.retriever_model.cuda(device=self.device)
        
    # Mean pooling
    @staticmethod
    def _mean_pooling(token_embeddings:torch.Tensor, mask) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings:torch.Tensor  = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def _embed_paragraphs(self, paragraphs:List[str]) -> np.ndarray:
        retriever_input = self.retriever_tokenizer.batch_encode_plus(paragraphs, padding=True, truncation=True, return_tensors='pt')
        if self.device:
            retriever_input = retriever_input.to(self.device)
        with torch.no_grad():
            retriever_output = self.retriever_model(**retriever_input)
            paragraph_embeddings = self._mean_pooling(retriever_output[0], retriever_input['attention_mask']).cpu().numpy()
        return paragraph_embeddings
    
    def _dense_retrieval(self, question:str, paragraphs:Union[List[str], np.ndarray], k:int=5):
        doc_embeds = paragraphs if isinstance(paragraphs, np.ndarray) else self._embed_paragraphs(paragraphs)
        query_embed = self._embed_paragraphs([question])
        scores = doc_embeds.dot(query_embed.squeeze())
        max_indices:List[int] = np.argsort(scores)[::-1][:k].tolist()
        return max_indices
    
    def _parse_to_graph(self, response:str):
        start_ents = False
        start_summary = False
        summary:List[str] = []
        line:str
        temp_graph = nx.Graph()
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.strip().startswith('Important ent'):
                start_ents = True
            elif line.strip().startswith('Relation summary'):
                start_summary = True
                start_ents = False
            else:
                if start_ents:
                    ent = line.strip('*+ ')
                    if ent.split('. ', 1)[0].isnumeric():
                        ent = ent.split('. ', 1)[-1]
                    temp_graph.add_node(ent, sum=[])
                elif start_summary and ':' in line:
                    sum = line.strip('*+ ')
                    if sum.split('. ', 1)[0].isnumeric():
                        sum = sum.split('. ', 1)[1]
                    summary.append(sum)

        rels = []
        for sum in summary:
            ents, rel = sum.split(':', 1)
            sid = len(rels)
            rels.append(rel.strip())
            if ents[0] == '(' and ents[-1] == ')':
                ents = ents[1:-1]
            ents = [ent.strip() for ent in ents.split(', ')]
            for ent in ents:
                if not temp_graph.has_node(ent):
                    temp_graph.add_node(ent, sum=[])
                temp_graph.nodes[ent]['sum'].append(sid)
            temp_graph.add_edges_from(itertools.combinations(ents, 2))

        return temp_graph, rels
    
    def index_text(self, paragraphs:List[str], context_type:str='novel'):
        results:List[Tuple[str, str]] = []
        for paragraph in paragraphs:
            # list_entity_prompt = f'''{context_type.upper()}:\n\n{paragraph}\n\nAbove is part of a {context_type}. First, list the important entities in the above passages that are relevant to most of the content. You may synthesis entities to avoid ambiguity. Don't give any explanation. Then, summarize the information in the above context for each of the important entities and try to include other important entities in each entity's summary if they are related. The two steps should be generated in the following format: "Important entities:\n1. Entity 1\n2. Entity 2\n...\nEntity summary:\n1. Entity 1: summary of Entity 1\n2. Entity 2: summary of Entity 2\n..."'''
            list_entity_prompt = f'''{context_type.upper()}:\n\n{paragraph}\n\nAbove is part of a {context_type}. First, list the important named entities in the above passages that are relevant to most of the content. You may synthesis entities to avoid ambiguity. Don't give any explanation. Then, find the closely related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above context. The two steps should be generated in the following format: "Important entities:\n1. Entity 1\n2. Entity 2\n...\n\nRelation summary:\n1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.\n2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.\n..."'''
            chat_response = self.llm_server(list_entity_prompt)
            results.append((paragraph, chat_response))
        return results
    
    def build_index(self, text_index:List[Tuple[str, str]]):
        all_graph = nx.DiGraph()
        all_summary:List[str] = []
        paragraphs = []
        pid2nodes = []
        for pid, (paragraph, response) in enumerate(text_index):
            paragraphs.append(paragraph)
            temp_graph, temp_summary = self._parse_to_graph(response)
            pid2nodes.append(list(temp_graph.nodes))
            s_offset = len(all_summary)
            for node, sum in temp_graph.nodes.data('sum'):
                if not all_graph.has_node(node):
                    all_graph.add_node(node, sum=[], pids=[])
                all_graph.nodes[node]['pids'].append(pid)
                all_graph.nodes[node]['sum'].extend([(sid + s_offset, pid) for sid in sum])
                
            # for head, tail, sids in temp_graph.edges.data('sum'):
            #     if not all_graph.has_edge(head, tail):
            #         all_graph.add_edge(head, tail, sum=[])
            #     all_graph.get_edge_data(head, tail)['sum'].extend([(sid + s_offset, pid) for sid in sids])
            all_graph.add_edges_from(temp_graph.edges)
            all_summary.extend(temp_summary)
        return DocIndex(all_graph, paragraphs, all_summary, self._embed_paragraphs(paragraphs), pid2nodes)
    
    def identify_noun_verb(self, query:str, n:int=10):
        query_entity_prompt = f'''Question: {query}\nYou need to answer the above question based on a given story. Before reading the story, identify important and unique noun and verb phrases in the question that you want to query from the story for useful information. All the phrases must appear in the question. Don't give any explanation. Generate your response in the following format:\n"Query noun phrases:\nthe first noun phrase, the second noun phrase, ...\n\nQuery verb phrases:\nthe first verb phrase, the second verb phrase, ...".'''
        chat_response = self.llm_server(query_entity_prompt, n=n, temperature=0.8, return_str=False)
        noun_phrases = Counter()
        verb_phrases = Counter()
        is_noun = False
        is_verb = False
        for choice in chat_response.choices:
            for line in choice.message.content.splitlines():
                line = line.strip(' .')
                if line:
                    if line.lower() == 'query noun phrases:':
                        is_noun = True
                    elif line.lower() == 'query verb phrases:':
                        is_noun = False
                        is_verb = True
                    else:
                        if is_noun:
                            noun_phrases.update(line.split(', '))
                        elif is_verb:
                            verb_phrases.update(line.split(', '))
                            break
        return [ent for ent, cnt in noun_phrases.most_common() if cnt >= n // 3 and (cal_rouge(ent.lower(), query.lower())['rouge-l']['f'] > 0 or ent.lower() in query.lower())], [kw for kw, cnt in verb_phrases.most_common() if cnt >= n // 3 and (cal_rouge(kw.lower(), query.lower())['rouge-l']['f'] > 0 or kw.lower() in query.lower())]
    
    def retrieve_node(self, doc_index:DocIndex, targets:List[str], k:int=5, threshold:float=0.5):
        nodes = list(doc_index.graph.nodes)
        nodes.sort()
        ret:List[List[str]] = []
        node_embeds = self._embed_paragraphs(nodes)
        node_embeds = node_embeds / np.expand_dims(np.linalg.norm(node_embeds, axis=1), axis=1)
        for target in targets:
            ent_embed = self._embed_paragraphs([target])
            ent_embed = ent_embed / np.linalg.norm(ent_embed)
            scores:np.ndarray = node_embeds.dot(ent_embed.squeeze())
            # if scores.max() < threshold:
            #     self.add_node(doc_index, target)
            #     if doc_index.graph.has_node(target):
            #         ret.append([target])
            #         nodes = list(doc_index.graph.nodes)
            #         nodes.sort()
            #         node_embeds = self._embed_paragraphs(nodes)
            #         node_embeds = node_embeds / np.expand_dims(np.linalg.norm(node_embeds, axis=1), axis=1)
            #         continue
            max_indices = np.argsort(scores)[::-1][:k]
            ret.append([nodes[i] for i in max_indices if scores[i] >= threshold] + [target])
        return ret

    def add_node(self, doc_index:DocIndex, node:str):
        pids = self._dense_retrieval(node, doc_index.paragraph_embs, 10)
        for pid in pids:
            important_ents = "\n".join(doc_index.pid2nodes[pid])
            response = self.llm_server(f'''Context:\n{doc_index.paragraphs[pid]}\n\nImportant entities:\n{important_ents}\n\nDoes the above context contain information about "{node}"? If yes, summarize the information in the above context for "{node}" and try to include other important entities in the summary if they are related. If no, simply reply "No". Generate your response in the following format: "Answer: summary of {node} or 'No.'"''')
            response = response.strip()
            if response.lower().startswith('answer: '):
                if response.lower().startswith('answer: no'):
                    continue
                summary = f"{node}: {response[8:]}"
                if not doc_index.graph.has_node(node):
                    doc_index.graph.add_node(node, sum=[])
                sid = len(doc_index.summary)
                doc_index.summary.append(summary)
                doc_index.graph.nodes[node]['sum'].append((sid, pid))
                doc_index.pid2nodes[pid].append(node)
                for other_ent in doc_index.pid2nodes[pid]:
                    other_ent_mention = other_ent
                    if '(' in other_ent_mention:
                        other_ent_mention = other_ent_mention.split('(')[0].strip()
                    if ',' in other_ent_mention:
                        other_ent_mention = other_ent_mention.split(',')[0].strip()
                    if other_ent != node and other_ent_mention.lower() in summary:
                        if not doc_index.graph.has_edge(node, other_ent):
                            doc_index.graph.add_edge(node, other_ent, sum=[])
                        doc_index.graph.get_edge_data(node, other_ent)['sum'].append((sid, pid))
            
    def retrieve_menu(self, mention_sets:List[List[str]], doc_index:DocIndex, query:str=None):
        pid2info = defaultdict(lambda: {'ents': [], 'sids': set()})
        sid2nodes = defaultdict(set)
        for mention_set in mention_sets:
            for mention in mention_set:
                if doc_index.graph.has_node(mention):
                    for pid in doc_index.graph.nodes[mention]['pids']:
                        pid2info[pid]['ents'].append(mention)
                    for sid, pid in doc_index.graph.nodes[mention]['sum']:
                        sid2nodes[sid].add(mention)
                        pid2info[pid]['sids'].add(sid)
                else:
                    p_scores = doc_index.bm25.get_scores([w.lower() for w in word_tokenize(mention)])
                    max_indices = np.argsort(p_scores)[::-1][:5]
                    for pid in max_indices:
                        if p_scores[pid] > 0:
                            pid2info[pid]['ents'].append(mention)
                
        pids = list(pid2info.keys())
        pids.sort()
        menu = []
        for pid in pids:
            passage_info = f"Passage {pid}:"
            passage_info += f"\nEntities: {str(pid2info[pid]['ents']).strip('[]')}"
            if pid2info[pid]['sids']:
                passage_info += '\nRelational information:'
                for sid in pid2info[pid]['sids']:
                    passage_info += f'\n{sid2nodes[sid]}: {doc_index.summary[sid]}'
            menu.append(passage_info)
        return '\n\n'.join(menu)
    
    def retrieval_passage(self, command:str, doc_index:DocIndex):
        thought = ''
        passage_numbers = []
        for line in command.splitlines():
            line = line.strip()
            if line.lower().startswith('explanation: '):
                thought = line
            elif line.lower().startswith('passage numbers: ') or line.lower().startswith('passages: '):
                line = line.strip('.')
                passage_numbers = [p.strip(',') for p in line.split(': ', 1)[1].strip('. ').split() if p.count('-') <= 0 and all([c.isnumeric() or c in ',-' for c in p])]
                passage_numbers = [int(p) if '-' not in p else [int(i) for i in p.split('-')] for p in passage_numbers]
        
        if passage_numbers:
            pids = set()
            for p in passage_numbers:
                if isinstance(p, int):
                    pids.add(p)
                else:
                    pids.update(range(p[0], p[1] + 1))
            pids = [pid for pid in pids if pid < len(doc_index.paragraphs)]
            pids.sort()
            
            return pids[:6]
        
    def parse_decision(self, decision:str):
        for line in decision.splitlines():
            line = line.strip(' .').lower()
            if line.lower().startswith('answer: '):
                if line.split(': ', 1)[1].lower().startswith('yes'):
                    return True
                if line.split(': ', 1)[1].lower().startswith('no'):
                    return False
    
                
    def main(self, query:str, index_file:str, log_file:str=None, r_tool:str='index'):
        # Step 1: Given a question, find the named entities and important keywords
        # Step 2: For named entities, check the indexed entities; for important keywords and un-recognized entities, use BM25 to match passages
        doc_index = self.build_index(read_json(index_file))
        log_info(log_file, 'query', query)
        
        if r_tool == 'index':
            # Step 1: identity entities of interest
            noun_phrases, verb_phrases = self.identify_noun_verb(query)
            while len(noun_phrases) == 0:
                noun_phrases, verb_phrases = self.identify_noun_verb(query)
            log_info(log_file, 'entity & keyword', noun_phrases)
        
            threshold = 0.5
            k = 10
            mention_sets = self.retrieve_node(doc_index, noun_phrases, k, threshold)
            mention_sets = [list(s) for s in set([frozenset(s) for s in mention_sets if s])]
            log_info(log_file, 'mention_sets', mention_sets)
            
            # Step 2: retrieve summary/original text
            menu = self.retrieve_menu(mention_sets, doc_index, query)
            log_info(log_file, 'menu', menu)
            passage_retrieve_prompt = f'''Question: {query}\n\nYou need to answer the above question based on a given story.\nBelow is a list of passages from the story with question-related entities, entity pairs and entity summaries contained in each passage.\n\n'''
            passage_retrieve_prompt += menu
            passage_retrieve_prompt += '''To answer the question, select 5 passages to retrieve passages from the original story for complete context. List the 5 passage numbers for passage retrieval. Write down your choice of passage numbers first and then your explanation. Your response should use the following format:\n"Passage numbers: the first passage number, the second passage number, ...\n\nExplanation: Your thought and reasoning in selecting passage numbers."\nDO NOT select more passages if you don't need to. DO NOT answer the question yet.'''

            retrieval_command = self.llm_server(passage_retrieve_prompt)
            log_info(log_file, 'retrieval_command', retrieval_command)
            pids = self.retrieval_passage(retrieval_command, doc_index)
            assert pids
            
        elif r_tool == 'dpr':
            pids = self._dense_retrieval(query, doc_index.paragraphs, 5)
            pids.sort()
        
        retrieval_result = ''.join([f'''Passage {pid}:\n{doc_index.paragraphs[pid]}\n\n''' for pid in pids])
        log_info(log_file, 'retrieval_result', retrieval_result)
        
        # Step 3: analyze retrieved info
        generation = self.llm_server(prompt_answer_template.format(question_type=self.dataset.question_type, article=retrieval_result, question=query, answer_format=self.dataset.answer_format))
        log_info(log_file, 'generation', generation)
        

    def self_rag(self, init_question:str, page_file:str, log_file:str=None):
        paragraphs = ['\n'.join(page) for page in read_json(page_file)]
        state = 'retrieve'
        context = ''
        generation = ''
        retrieved_pids = []
        query = init_question
        log_info(log_file, 'query', query)
        repeat = 0
        step = 1
        while True:
            if step >= 6:
                generation = self.llm_server(prompt_answer_template.format(question_type=dataset.question_type, article=context, question=init_question, answer_format=dataset.answer_format))
                log_info(log_file, 'generation', generation)
                break
            if state == 'retrieve':
                score_template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
                Here is the user question: \n\n {question} \n\n
                Here is the retrieved document: \n\n {document} \n\n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                The goal is to filter out erroneous retrievals. \n
                Provide the binary score as a JSON with a single key 'score' and no premable or explaination. \n
                DO NOT answer the question yet."""

                log_info(log_file, 'retrieval_command', query)
                pids = longdoc._dense_retrieval(query, paragraphs, 5)
                log_info(log_file, 'candidate_pids', pids)
                new_retrieved_pids = []
                for pid in pids:
                    doc_txt = paragraphs[pid]
                    score_response = self.llm_server(score_template.format(document=doc_txt, question=query))
                    if 'yes' in score_response.lower():
                        new_retrieved_pids.append(pid)
                if new_retrieved_pids:
                    log_info(log_file, 'relevant_pids', new_retrieved_pids)
                    retrieved_pids.extend([pid for pid in new_retrieved_pids if pid not in retrieved_pids])
                    retrieved_pids = retrieved_pids[-6:]
                    log_info(log_file, 'retrieved_pids', retrieved_pids)
                    context = '\n\n'.join([f'Passage {pid}:\n{paragraphs[pid]}' for pid in sorted(retrieved_pids)])
                    log_info(log_file, 'retrieval_result', context)
                    repeat = 0
                    state = 'generate'
                else:
                    state = 'transform_query'
                
            if state == 'generate':
                ### Generate
                prompt_summarize_template = '''
                Read the following article and a question.

                Article:
                {article}

                Question:
                {question}
                
                Generate a statement to summarize useful information in the article to the question.
                DO NOT answer the question yet.
                '''
                generation = self.llm_server(prompt_summarize_template.format(article=context, question=init_question), temperature=0.7)
                log_info(log_file, 'temp_generation', generation)
                
                ### Hallucination Grader 
                hallucination_template="""You are a grader assessing whether a statement is grounded in / supported by a set of facts. \n 
                Here are the facts:
                \n ------- \n
                {documents} 
                \n ------- \n
                Here is the statement: {generation}
                Give a binary score 'yes' or 'no' score to indicate whether the statement is grounded in / supported by a set of facts. \n
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

                hallucination_response = self.llm_server(hallucination_template.format(documents=context, generation=generation))
                log_info(log_file, 'hallucination_response', hallucination_response)
                
                if 'yes' in hallucination_response.lower() or repeat >= 5:
                    if repeat >= 5:
                        log_info(log_file, 'error', 'skip reliable check')
                        
                    ### Answer Grader 
                    answer_grade_template="""You are a grader assessing whether a statement contains adequate information to resolve a question. \n 
                    Here is the question: \n\n{question} \n\n
                    Here is the statement:
                    \n ------- \n
                    {generation} 
                    \n ------- \n
                    Give a binary score 'yes' or 'no' to indicate whether the statement contains adequate information to resolve a question. \n
                    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

                    answer_grade_response = self.llm_server(answer_grade_template.format(question=init_question, generation=generation))
                    log_info(log_file, 'answer_grade_response', answer_grade_response)
                    if 'yes' in answer_grade_response.lower():
                        generation = self.llm_server(prompt_answer_template.format(question_type=dataset.question_type, article=context, question=init_question, answer_format=dataset.answer_format))
                        log_info(log_file, 'generation', generation)
                        break
                    else:
                        state = 'transform_query'
                else:
                    state = 'generate'
                    repeat += 1
            
            if state == 'transform_query':
                ### Question Re-writer
                re_write_template="""You are a query re-writer that converts an input query to a better version that is optimized \n 
                for vectorstore retrieval. Based on the initial question, the current query and the current generated response, formulate an improved query. \n
                Here is the initial question: \n\n {question} \n\n
                Here is the current query: \n\n {query} \n\n
                Here is the current response: \n\n {generation} \n\n
                Improved question with no preamble: \n """

                re_write_response = self.llm_server(re_write_template.format(question=init_question, query=query, generation=generation))
                query = re_write_response
                state = 'retrieve'
                step += 1
                

if __name__ == '__main__':
    
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='quality', choices=['narrativeqa', 'quality', 'musique'])
    parser.add_argument('--r_tool', type=str, default='index', choices=['dpr', 'index', 'gist'])
    parser.add_argument('--reasoning_style', type=str, default='p', choices=['s', 'p'])
    args = parser.parse_args()

    context_type = 'novel'

    task_name = args.task
    r_tool = args.r_tool
    reasoning_style = args.reasoning_style
    
    if task_name == 'narrativeqa':
        dataset = NarrativeQADataset()
    elif task_name == 'musique':
        dataset = MuSiQueDataset()
    else:
        dataset = QualityDataset("mistralai/Mistral-7B-Instruct-v0.2", 'dev')
        
    longdoc = LongDoc(dataset, llm_name="mistralai/Mistral-7B-Instruct-v0.2", device='cpu')
    reading_agent = ReadingAgent(dataset)
    
    for task_i in range(0, 10):
        print(f'{task_i} start')
        
        index_file = os.path.join(dataset.data_dir, f'rel_index_{task_i}.json')
        page_file = os.path.join(dataset.data_dir, f'pages_{task_i}.json')
        gist_file = os.path.join(dataset.data_dir, f'gist_{task_i}.json')
        log_file = os.path.join(dataset.data_dir, f'generation_{r_tool}_{reasoning_style}_{task_i}.jsonl')
        questions, _ = dataset.get_questions_and_answers(dataset.data[task_i])
        for qid, query in enumerate(questions):
            if r_tool == 'gist':
                reading_agent.main(query, page_file, gist_file, log_file, reasoning_style)
            else:
                if reasoning_style == 'p':
                    longdoc.main(query, index_file, log_file, r_tool)
                else:
                    longdoc.self_rag(query, page_file, log_file)
            
        print(f'{task_i} end')

        