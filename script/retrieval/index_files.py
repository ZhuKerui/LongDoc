from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Any, Union
import torch
from nltk import sent_tokenize
from openai import OpenAI
import networkx as nx
from tqdm import tqdm
import json
import numpy as np
import itertools
from collections import defaultdict, Counter
from rouge_metric import PyRouge


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

class DocIndex:
    def __init__(self, graph:nx.DiGraph, paragraphs:List[str], summary:List[str], paragraph_embs:np.ndarray, pid2nodes:List[List[str]]) -> None:
        self.graph = graph
        self.paragraphs = paragraphs
        self.summary = summary
        self.paragraph_embs = paragraph_embs
        self.pid2nodes = pid2nodes
    
    
class DocSplit:
    def split_trec(text:str):
        lines = text.splitlines()
        return ['\n'.join(lines[i * 2 : i * 2 + 1]) for i in range(len(lines) // 2)]

    def split_triviaqa(text:str):
        lines = text.splitlines()
        paragraphs = []
        paragraph = []
        lid = 0
        while lid < len(lines):
            paragraph.append(lines[lid])
            if lines[lid] == 'Answer:':
                lid += 1
                paragraph.append(lines[lid])
                paragraphs.append('\n'.join(paragraph))
                paragraph.clear()
            lid += 1
        return paragraphs

    def split_samsum(text:str):
        paragraphs = []
        paragraph = []
        for line in text.splitlines():
            paragraph.append(line)
            if line.startswith('Summary: '):
                paragraphs.append('\n'.join(paragraph))
                paragraph.clear()
        return paragraphs

    paragraph_sep_map = {
        'qasper': '\n', 
        'multifieldqa_zh': '\n', 
        'qmsum': '\n', 
        'multi_news': '\n', 
        'vcsum': '\n', 
        'trec': (split_trec, '\n'), 
        'triviaqa': (split_triviaqa, '\n'), 
        'samsum': (split_samsum, '\n'), 
    }
    
    def __init__(self, llm_name:str) -> None:
        self.llm_name = llm_name
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        
    def _append_paragraph(self, paragraphs:list, tokenized_p:List[str]):
        paragraph = self.llm_tokenizer.decode(tokenized_p)
        paragraphs.append(paragraph)
        tokenized_p.clear()
        
    def get_task_paragraph_sep(self, task_name:str):
        sep = self.paragraph_sep_map.get(task_name, '\n\n')
        if not isinstance(sep, str):
            func, sep = sep
        return sep
    
    def split_context_to_paragraphs(self, context:str, task_name:str):
        sep = self.paragraph_sep_map.get(task_name, '\n\n')
        if isinstance(sep, str):
            return context.split(sep)
        else:
            func, sep = self.paragraph_sep_map[task_name]
            return func(context)
        
    def split_single_paragraph(self, text:str, paragraph_size:int=300, is_natural_language:bool=True):
        splited_paragraphs:List[str] = []
        splited_paragraph = []
        sentences:List[str] = sent_tokenize(text) if is_natural_language else text.split('\n')
        for sent in sentences:
            tokenized_s = self.llm_tokenizer.encode(sent)[1:]
            if len(tokenized_s) <= paragraph_size:
                if len(splited_paragraph) + len(tokenized_s) > paragraph_size:
                    self._append_paragraph(splited_paragraphs, splited_paragraph)
                splited_paragraph.extend(tokenized_s)
            else:
                if splited_paragraph:
                    self._append_paragraph(splited_paragraphs, splited_paragraph)
                chunk_size = (len(tokenized_s) - 1) // paragraph_size + 1
                for i in range(chunk_size - 1):
                    self._append_paragraph(splited_paragraphs, tokenized_s[i * paragraph_size: (i+1) * paragraph_size])
                splited_paragraph = tokenized_s[(chunk_size - 1) * paragraph_size:]
        return splited_paragraphs, splited_paragraph
        
    def split_paragraphs(self, text:str, task_name:str, paragraph_size:int=300):
        reformated_paragraphs:List[str] = []
        completion_labels:List[bool] = []
        reformated_paragraph = []
        
        paragraph_sep = self.get_task_paragraph_sep(task_name)
        paragraphs = text.split(paragraph_sep)
        for p in paragraphs:
            tokenized_p = self.llm_tokenizer.encode(p + paragraph_sep)[1:]
            if len(tokenized_p) <= paragraph_size:
                if len(reformated_paragraph) + len(tokenized_p) > paragraph_size:
                    self._append_paragraph(reformated_paragraphs, reformated_paragraph)
                    completion_labels.append(True)
                reformated_paragraph.extend(tokenized_p)
            else:
                if reformated_paragraph:
                    self._append_paragraph(reformated_paragraphs, reformated_paragraph)
                    completion_labels.append(True)
                splited_paragraphs, splited_paragraph = self.split_single_paragraph(p, paragraph_size)
                reformated_paragraphs.extend(splited_paragraphs)
                completion_labels.extend([False] * len(splited_paragraphs))
                reformated_paragraph = splited_paragraph
                
        if reformated_paragraph:
            self._append_paragraph(reformated_paragraphs, reformated_paragraph)
            completion_labels.append(True)
        return reformated_paragraphs, completion_labels
    
    
class LongDoc:
    
    def __init__(self, retriever_model_name:str='facebook/contriever', llm_name:str='meta-llama/Llama-2-7b-hf', device='cpu') -> None:
        self.llm_name = llm_name
        if 'gpt' in self.llm_name:
            self.doc_split = DocSplit('meta-llama/Llama-2-7b-hf')
            self.llm_server = OpenAI()
        else:
            self.doc_split = DocSplit(self.llm_name)
            
            # Set OpenAI's API key and API base to use vLLM's API server.
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8000/v1"
            self.llm_server = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
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
        ents:List[str] = []
        line:str
        temp_graph = nx.DiGraph()
        for line in response.splitlines():
            if not line:
                continue
            if line.strip().startswith('Important ent'):
                start_ents = True
            elif line.strip().startswith('Entity summary'):
                start_summary = True
                start_ents = False
            else:
                if start_summary and ':' in line:
                    summary.append(line.strip('*+ '))

        summary = [sum.split('. ', 1)[1] if sum.startswith(f'{sid+1}.') else sum for sid, sum in enumerate(summary)]
        summary = [sum.split(': ', 1)[1].replace(' -', ':', 1) if sum.startswith(f'Entity {sid+1}:') else sum for sid, sum in enumerate(summary)]
        for sid, sum in enumerate(summary):
            ent, rel = sum.split(':', 1)
            temp_graph.add_node(ent.strip(), sum=sid)

        for sid, sum in enumerate(summary):
            ent, rel = sum.split(':', 1)
            rel = rel.lower()
            ent = ent.strip()
            for other_ent in temp_graph.nodes:
                other_ent_mention = other_ent
                if '(' in other_ent_mention:
                    other_ent_mention = other_ent_mention.split('(')[0].strip()
                if ',' in other_ent_mention:
                    other_ent_mention = other_ent_mention.split(',')[0].strip()
                if other_ent != ent and other_ent_mention.lower() in rel:
                    if not temp_graph.has_edge(ent, other_ent):
                        temp_graph.add_edge(ent, other_ent, sum=[])
                    temp_graph.get_edge_data(ent, other_ent)['sum'].append(sid)
        return temp_graph, summary
    
    def _call_llm(self, prompt:str, n:int=1, temperature:float=0.0):
        chat_response = self.llm_server.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            n=n,
            temperature=temperature,
        )
        return chat_response
    
    def index_text(self, paragraphs:List[str], context_type:str='novel'):
        results:List[Tuple[str, str]] = []
        for paragraph in tqdm(paragraphs):
            list_entity_prompt = f'''{context_type.upper()}:\n\n{paragraph}\n\nAbove is part of a {context_type}. First, list the important entities in the above passages that are relevant to most of the content. You may synthesis entities to avoid ambiguity. Don't give any explanation. Then, summarize the information in the above context for each of the important entities and try to include other important entities in each entity's summary if they are related. The two steps should be generated in the following format: "Important entities:\n1. Entity 1\n2. Entity 2\n...\nEntity summary:\n1. Entity 1: summary of Entity 1\n2. Entity 2: summary of Entity 2\n..."'''
            chat_response = self._call_llm(list_entity_prompt)
            results.append((paragraph, chat_response.choices[0].message.content))
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
            for node, sid in temp_graph.nodes.data('sum'):
                if not all_graph.has_node(node):
                    all_graph.add_node(node, sum=[])
                all_graph.nodes[node]['sum'].append((sid + s_offset, pid))
            for head, tail, sids in temp_graph.edges.data('sum'):
                if not all_graph.has_edge(head, tail):
                    all_graph.add_edge(head, tail, sum=[])
                all_graph.get_edge_data(head, tail)['sum'].extend([(sid + s_offset, pid) for sid in sids])
            all_summary.extend(temp_summary)
        return DocIndex(all_graph, paragraphs, all_summary, self._embed_paragraphs(paragraphs), pid2nodes)
    
    def identify_noun_verb(self, query:str, n:int=10):
        query_entity_prompt = f'''Question: {query}\nYou need to answer the above question based on a given story. Before reading the story, identify important and unique noun and verb phrases in the question that you want to query from the story for useful information. All the phrases must appear in the question. Don't give any explanation. Generate your response in the following format:\n"Query noun phrases:\nthe first noun phrase, the second noun phrase, ...\n\nQuery verb phrases:\nthe first verb phrase, the second verb phrase, ...".'''
        chat_response = self._call_llm(query_entity_prompt, n=n, temperature=0.8)
        # print(chat_response.choices[0].message.content)
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
            ret.append([nodes[i] for i in max_indices if scores[i] >= threshold])
        return ret

    def add_node(self, doc_index:DocIndex, node:str):
        pids = self._dense_retrieval(node, doc_index.paragraph_embs, 10)
        for pid in pids:
            important_ents = "\n".join(doc_index.pid2nodes[pid])
            response = self._call_llm(f'''Context:\n{doc_index.paragraphs[pid]}\n\nImportant entities:\n{important_ents}\n\nDoes the above context contain information about "{node}"? If yes, summarize the information in the above context for "{node}" and try to include other important entities in the summary if they are related. If no, simply reply "No". Generate your response in the following format: "Answer: summary of {node} or 'No.'"''').choices[0].message.content
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
        pair2sids = set()
        for mention_set in mention_sets:
            for mention in mention_set:
                pair2sids.update([((mention, ), sid, pid) for sid, pid in doc_index.graph.nodes[mention]['sum']])
                for pair in doc_index.graph.edges(mention):
                    pair2sids.update([((mention, ), sid, pid) for sid, pid in doc_index.graph.get_edge_data(*pair)['sum']])
        for pairs in itertools.combinations(mention_sets, 2):
            for m_node, a_node in itertools.product(*pairs):
                if doc_index.graph.has_edge(m_node, a_node):
                    pair2sids.update([((m_node, a_node), sid, pid) for sid, pid in doc_index.graph.get_edge_data(m_node, a_node)['sum']])
                if doc_index.graph.has_edge(a_node, m_node):
                    pair2sids.update([((a_node, m_node), sid, pid) for sid, pid in doc_index.graph.get_edge_data(a_node, m_node)['sum']])
        pair2sids = list(pair2sids)
        pair2sids.sort(key=lambda x: x[2])
        sids = list({sid for pair, sid, pid in pair2sids})
        sids = [sids[idx] for idx in self._dense_retrieval(query, [doc_index.summary[sid] for sid in sids], 10)]
        pid2pairs = defaultdict(set)
        pid2sids = defaultdict(set)
        for pair, sid, pid in pair2sids:
            pid2pairs[pid].add(frozenset(pair))
            if sid in sids:
                pid2sids[pid].add(sid)
        menu = []
        for pid, pairs in pid2pairs.items():
            important_ents = []
            important_pairs = []
            for pair in pairs:
                if len(pair) == 2:
                    important_pairs.append(tuple(pair))
                else:
                    important_ents.append(tuple(pair)[0])
            passage_info = f"Passage {pid}:"
            if important_ents:
                passage_info += f"\nImportant entities: {str(important_ents).strip('[]')}"
            if important_pairs:
                passage_info += f"\nImportant pairs: {str(important_pairs).strip('[]')}"
            passage_info += "\nImportant entity summaries:\n"
            passage_info += '\n'.join([doc_index.summary[sid] for sid in pid2sids[pid]])
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
            expanded_passage_numbers = set()
            for p in passage_numbers:
                if isinstance(p, int):
                    expanded_passage_numbers.add(p)
                else:
                    expanded_passage_numbers.update(range(p[0], p[1] + 1))
            expanded_passage_numbers = list(expanded_passage_numbers)
            expanded_passage_numbers.sort()
            retrieval_result = ''
            for pid in expanded_passage_numbers:
                if pid < len(doc_index.paragraphs):
                    retrieval_result += f'''Passage {pid}:\n{doc_index.paragraphs[pid]}\n\n'''
            return retrieval_result
        
    def parse_decision(self, decision:str):
        for line in decision.splitlines():
            line = line.strip(' .').lower()
            if line.lower().startswith('answer: '):
                if line.split(': ', 1)[1].lower().startswith('yes'):
                    return True
                if line.split(': ', 1)[1].lower().startswith('no'):
                    return False
    
    def log_info(self, log_file:str, tag:str, info:Any):
        if log_file is not None:
            with open(log_file, 'a') as f_out:
                f_out.write(json.dumps([tag, info]))
                f_out.write('\n')
                
    def main(self, query:str, index_file:str, log_file:str=None, r_tool:str='index'):
        doc_index = self.build_index(read_json(index_file))
        self.log_info(log_file, 'query', query)
        
        if r_tool == 'index':
            # Step 1: identity entities of interest
            noun_phrases, verb_phrases = self.identify_noun_verb(query)
            while len(noun_phrases) == 0:
                noun_phrases, verb_phrases = self.identify_noun_verb(query)
            self.log_info(log_file, 'entity & keyword', noun_phrases)
        
            threshold = 0.5
            k = 10
            mention_sets = self.retrieve_node(doc_index, noun_phrases, k, threshold)
            mention_sets = [list(s) for s in set([frozenset(s) for s in mention_sets if s])]
            self.log_info(log_file, 'mention_sets', mention_sets)
            
            # Step 2: retrieve summary/original text
            menu = self.retrieve_menu(mention_sets, doc_index, query)
            self.log_info(log_file, 'menu', menu)
            passage_retrieve_prompt = f'''Question: {query}\n\nYou need to answer the above question based on a given story.\nBelow is a list of passages from the story with question-related entities, entity pairs and entity summaries contained in each passage.\n\n'''
            passage_retrieve_prompt += menu
            passage_retrieve_prompt += '''To answer the question, select 1-6 passages for more information. List the passage numbers for passage retrieval. Write down your choice of passage numbers first and then your explanation. Your response should use the following format:\n"Passage numbers: the first passage number, the second passage number, ...\n\nExplanation: Your thought and reasoning in selecting passage numbers."'''

            # passage_retrieve_prompt += '''To answer the question, select 5 passages for more information. You may list the passage numbers for single passage retrieval (e.g., "1, 3, 4" for passage 1, 3 and 4) or passage spans for continuous passages (e.g., "1-3" for passage 1, 2, and 3). Write down your thought first and then generate your choice of passage numbers. Your response should use the following format:\n"Thought: Your thought and reasoning in selecting retrieval type and passage numbers.\n\nPassage numbers: first passage number or span, second passage number or span, ..."'''
            # passage_retrieve_prompt += '''To answer the question, select 5 passages and retrieve the "original text" or "summary" of each passage for more information. You may list the passage numbers for single passage retrieval (e.g., "1, 3, 4" for passage 1, 3 and 4) or passage spans for continuous passages (e.g., "1-3" for passage 1, 2, and 3). Write down your thought first and then generate your choice of retrieval type and passage numbers. Your response should use the following format:\n"Thought: Your thought and reasoning in selecting retrieval type and passage numbers.\nRetrieval type: summary/original text\nPassage numbers: first passage number or span, second passage number or span, ..."'''
            retrieval_result = None
            fail_cnt = -1
            while retrieval_result is None:
                fail_cnt += 1
                assert fail_cnt < 5
                retrieval_command = self._call_llm(passage_retrieve_prompt).choices[0].message.content
                self.log_info(log_file, 'retrieval_command', retrieval_command)
                retrieval_result = self.retrieval_passage(retrieval_command, doc_index)
            # retrieval_result, retrieval_type = retrieval_result
            
        elif r_tool == 'dpr':
            pids = self._dense_retrieval(query, doc_index.paragraphs, 5)
            retrieval_result = ''.join([f'''Passage {pid}:\n{doc_index.paragraphs[pid]}\n\n''' for pid in pids])
            # retrieval_type = 'original text'
            
        self.log_info(log_file, 'retrieval_result', retrieval_result)
        
        # Step 3: analyze retrieved info
        # context_type = 'passages' if retrieval_type == 'original text' else 'entity summaries in passages'
        context_type = 'passages'
        # analyze_retrieve_prompt = f'''Question: {query}\n\nYou need to answer the above question based on a given story. Below are some selected {context_type}.\n\n'''
        # analyze_retrieve_prompt += retrieval_result
        # analyze_retrieve_prompt += '''Now, summarize any useful information from the above passages. Generate your response in the following format:\n"Summary: the summary of current useful information".'''
        
        analyze_retrieve_prompt = '''
Read the following passages and answer a multiple choice question.
For example, if (C) is correct, answer with \"Answer: (C) ...\"

Article:
{}

Question:
{}

'''.format(retrieval_result, query)
        current_summary = self._call_llm(analyze_retrieve_prompt).choices[0].message.content
        self.log_info(log_file, 'current_summary', current_summary)
        
        # # Step 4: continue searching or start answering
        # decision_prompt = f'''Question: {query}\n\nYou need to answer the above question based on a given story.\nBelow is a summary of currently retrieved information.\n\n'''
        # decision_prompt += current_summary
        # decision_prompt += '''\n\nIs the information enough for answering the question? Write down your thought first and then generate your answer. Your response should use the following format:\n"Thought: Your thought and reasoning in judging whether the information is enough.\nAnswer: Yes/No".'''
        # enough_information = None
        # fail_cnt = -1
        # while enough_information is None:
        #     fail_cnt += 1
        #     assert fail_cnt < 5
        #     decision = self._call_llm(decision_prompt).choices[0].message.content
        #     enough_information = self.parse_decision(decision)
        # self.log_info(log_file, 'decision', decision)
        # return enough_information


if __name__ == '__main__':
    
    from datasets import load_dataset
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='quality', choices=['narrativeqa', 'quality'])
    parser.add_argument('--r_tool', type=str, default='index', choices=['dpr', 'index'])
    args = parser.parse_args()
    
    longdoc = LongDoc(llm_name="mistralai/Mistral-7B-Instruct-v0.2", device='cuda:0')

    context_type = 'novel'
    ent_num = 10

    task_name = args.task
    r_tool = args.r_tool
    if not os.path.exists(task_name):
        os.mkdir(task_name)
    
    if task_name == 'narrativeqa':
        dataset = load_dataset('THUDM/LongBench', task_name, split='test')
        process_sample = lambda sample: (sample['context'], [sample['input']])
    else:
        dataset = read_jsonline('../../data/QuALITY/QuALITY.v1.0.1.htmlstripped.train')
        process_sample = lambda sample: (sample['article'], ['\n'.join([q['question']] + [f'{c} {o}' for c, o in zip(['(A)', '(B)', '(C)', '(D)'], q['options'])]) for q in sample['questions']])

    for task_i in range(0, 10):
        
        print(f'{task_i} start')
        context, queries = process_sample(dataset[task_i])
        index_file = f'{task_name}/response_{task_i}.json'
        
        if not os.path.exists(index_file):
            paragraphs, completion_labels = longdoc.doc_split.split_paragraphs(context, task_name, 400)
            write_json(index_file, longdoc.index_text(paragraphs))
        
        for qid, query in enumerate(queries):
            # if qid != 4:
            #     continue
            one_time_pass = longdoc.main(query, index_file, f'{task_name}/response_{r_tool}_{task_i}_log.jsonl', r_tool)
            # if not one_time_pass:
            #     print(task_name, task_i, qid)
        print(f'{task_i} end')

        