import copy

from .prompt import GeneralPrompt, LongDocPrompt, ReadingAgentPrompt
from .data import *

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


class ReadingAgent:
    def __init__(self, dataset:Dataset, llm:str | LLM="mistralai/Mistral-7B-Instruct-v0.2", model_type:str = 'gpt') -> None:
        self.llm_server = LLMServer(llm)
        self.model_type = model_type
        self.dataset = dataset
    

    def gisting(self, example, pages, verbose=True):
        article = self.dataset.get_article(example)
        word_count = Dataset.count_words(article)

        shortened_pages = []
        for i, page in enumerate(pages):
            prompt = GeneralPrompt.shorten('\n'.join(page))
            response = self.llm_server(prompt)[0]
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
    
    
    def get_page_ids(self, response:str, max_page_num:int):
        start = response.lower().index('look up Page'.lower()) + len('look up Page')
        page_ids_str = response[start:].split('to', 1)[0].split()
        page_ids = []
        for p in page_ids_str:
            if p.strip(',.[]').isnumeric():
                page_id = int(p.strip(',.[]'))
                if page_id >= 0 and page_id <= max_page_num:
                    page_ids.append(page_id)
        return page_ids

    def main(self, query:str, page_file:str, gist_file:str, log_file:str=None, lookup_method:str='s'):
        pages = read_json(page_file)
        shortened_pages = read_json(gist_file)['shortened_pages']
        expanded_shortened_pages = shortened_pages[:]
        
        log_info(log_file, 'query', query)
        
        if lookup_method == 'p':
            menu = '\n'.join([f"<Page {i}>\n{shortened_text}" for i, shortened_text in enumerate(shortened_pages)])
            log_info(log_file, 'menu', menu)

            prompt_lookup = ReadingAgentPrompt.parallel_lookup(menu, query)
            # log_info(log_file, 'lookup_prompt', prompt_lookup)

            page_ids = []
            response = self.llm_server(prompt=prompt_lookup)[0].strip()
            log_info(log_file, 'retrieval_command', response)
            
            if 'look up Page'.lower() in response.lower():
                page_ids = self.get_page_ids(response, len(pages) - 1)

            # Memory expansion after look-up, replacing the target shortened page with the original page
            if len(page_ids) > 0:
                for page_id in page_ids:
                    expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])
        else:
            retrieved_passage_nums = []
            for _ in range(7):
                menu = '\n'.join([f"<Page {i}>\n{shortened_text}" for i, shortened_text in enumerate(expanded_shortened_pages)])
                log_info(log_file, 'menu', menu)
                
                # prompt_lookup = prompt_sequential_lookup_template.format(menu, query, ', '.join(map(str, retrieved_passage_nums)))
                prompt_lookup = ReadingAgentPrompt.sequential_lookup(menu, query.split('\n')[0], ', '.join(map(str, retrieved_passage_nums)))
                # log_info(log_file, 'lookup_prompt', prompt_lookup)
                
                response = self.llm_server(prompt=prompt_lookup)[0].strip()
                log_info(log_file, 'retrieval_command', response)
                
                # page_id:List[str] = re.findall(r'page \d+', response.lower())
                
                if 'STOP' in response or 'look up Page'.lower() not in response.lower():
                    log_info(log_file, 're-read complete', 'yes')
                    break
                else:
                    # Memory expansion after look-up, replacing the target shortened page with the original page
                    if len(retrieved_passage_nums) == 6:
                        log_info(log_file, 're-read complete', 'no')
                    else:
                        page_ids = self.get_page_ids(response, len(pages) - 1)
                        new_page_id_added = False
                        for page_id in page_ids:
                            if page_id not in retrieved_passage_nums:
                                expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])
                                retrieved_passage_nums.append(page_id)
                                log_info(log_file, 'retrieved_pids', retrieved_passage_nums)
                                new_page_id_added = True
                                break
                        if not new_page_id_added:
                            log_info(log_file, 're-read complete', 'yes')
                            break

        expanded_shortened_article = '\n'.join(expanded_shortened_pages)
        log_info(log_file, 'retrieval_result', expanded_shortened_article)

        response = self.llm_server(prompt=GeneralPrompt.answer(self.dataset.question_type, expanded_shortened_article, query, self.dataset.answer_format))[0]
        response = response.strip()
        log_info(log_file, 'generation', response)
    
    
class LongDoc:
    
    def __init__(self, retriever:Retriever, llm:str | LLM="mistralai/Mistral-7B-Instruct-v0.2") -> None:
        # self.dataset = dataset
        if llm:
            self.llm_server = LLMServer(llm)
        self.retriever = retriever
        
    def retrieve_descriptions(self, notes:List[ChunkInfo], relation_graph:nx.Graph, target_ents:List[str], r_num:int=1):
        ent2pids = defaultdict(list)
        pid:int
        for pid, note in enumerate(notes):
            for ent in note.ent_descriptions.keys():
                ent2pids[ent].append(pid)
        all_ents = list(ent2pids.keys())
        target_ents_emb:np.ndarray = self.retriever.embed_paragraphs(target_ents, True)
        refer_ents_emb:np.ndarray = self.retriever.embed_paragraphs(all_ents, True)
        ent_map = LongDocPrompt.match_entities(target_ents, all_ents, target_ents_emb, refer_ents_emb, 5)
        prev_ent_descriptions:Dict[int, Dict[str, str]] = defaultdict(dict)
        for _, mentions in ent_map.items():
            for mention in mentions:
                for pid in ent2pids[mention][-r_num:]:
                    prev_ent_descriptions[pid][mention] = notes[pid].ent_descriptions[mention]
        prev_relation_descriptions:Dict[int, List[Tuple[List[str], str]]] = defaultdict(list)
        all_nodes = list(relation_graph.nodes)
        refer_ents_emb:np.ndarray = self.retriever.embed_paragraphs(all_nodes, True)
        node_map = LongDocPrompt.match_entities(target_ents, all_nodes, target_ents_emb, refer_ents_emb, 5)
        sub_nodes = set()
        for ent, mentions in node_map.items():
            sub_nodes.update(mentions)
        sub_graph:nx.Graph = relation_graph.subgraph(sub_nodes)
        visited_relation_descriptions = set()
        for e1, e2, pids in sub_graph.edges.data('pids'):
            for pid in pids[-r_num:]:
                for rid, (related_ents, relation_description) in enumerate(notes[pid].relation_descriptions):
                    if e1 in related_ents and e2 in related_ents and (pid, rid) not in visited_relation_descriptions:
                        prev_relation_descriptions[pid].append((related_ents, relation_description))
                        visited_relation_descriptions.add((pid, rid))
        return prev_ent_descriptions, prev_relation_descriptions

    def index_text(self, paragraphs:List[str], w_note:bool=True, r_num:int=1):
        results:List[ChunkInfo] = []
        relation_graph = nx.Graph()
        for cur_pid, paragraph in enumerate(tqdm(paragraphs)):
            chunk_info = ChunkInfo(cur_pid, paragraph)
            if results and w_note:
                summary_recap = {pid: results[pid].summary for pid in range(max(len(results)-r_num, 0), len(results))}
                chunk_info.prev_summaries = summary_recap
                list_entity_prompt = LongDocPrompt.list_entity_w_note(chunk_info.recap_str, paragraph)
            else:
                list_entity_prompt = LongDocPrompt.list_entity(paragraph)
            # Extract important entities
            chat_response = self.llm_server(list_entity_prompt, 5, 0.7)[0]
            chunk_info.important_ents = LongDocPrompt.parse_entities(chat_response, lambda x: self.retriever.embed_paragraphs(x, True))
            
            # Generate entity description, summary, relation description
            if results and w_note:
                chunk_info.prev_ent_descriptions, chunk_info.prev_relation_descriptions = self.retrieve_descriptions(results, relation_graph, chunk_info.important_ents, r_num)
                recap_str = chunk_info.recap_str
                ent_description_prompt = LongDocPrompt.ent_description_w_note(recap_str, paragraph, chunk_info.important_ents)
                summary_prompt = LongDocPrompt.shorten_w_note(recap_str, paragraph)
                relation_description_prompt = LongDocPrompt.relation_description_w_note(recap_str, paragraph, chunk_info.important_ents)
            else:
                ent_description_prompt = LongDocPrompt.ent_description(paragraph, chunk_info.important_ents)
                summary_prompt = LongDocPrompt.shorten(paragraph)
                relation_description_prompt = LongDocPrompt.relation_description(paragraph, chunk_info.important_ents)
            
            ent_description, relation_description, summary = self.llm_server([ent_description_prompt, relation_description_prompt, summary_prompt])
            ent_description, relation_description, summary = ent_description[0], relation_description[0], summary[0]
            chunk_info.summary = summary
            chunk_info.ent_descriptions = LongDocPrompt.parse_ent_description(ent_description, chunk_info.important_ents)
            chunk_info.relation_descriptions = LongDocPrompt.parse_relation_description(relation_description, chunk_info.important_ents)
            results.append(chunk_info)
            for related_ents, _ in chunk_info.relation_descriptions:
                for ent1, ent2 in itertools.combinations(related_ents, 2):
                    if not relation_graph.has_edge(ent1, ent2):
                        relation_graph.add_edge(ent1, ent2, pids=[])
                    temp_pids:List[int] = relation_graph.get_edge_data(ent1, ent2)['pids']
                    if cur_pid not in temp_pids:
                        temp_pids.append(cur_pid)
        return results
    
    
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
    
    llm = LLM()
    retriever = Retriever()
    # llm = 'mistralai/Mistral-7B-Instruct-v0.2'
    
    if task_name == 'narrativeqa':
        dataset = NarrativeQADataset(llm)
    elif task_name == 'musique':
        dataset = MuSiQueDataset(llm)
    else:
        dataset = QualityDataset(llm, 'dev')
        
    longdoc = LongDoc(dataset, retriever, llm)
    reading_agent = ReadingAgent(dataset, llm)
    
    for task_i in range(0, 10):
        print(f'{task_i} start')
        
        index_file = os.path.join(dataset.data_dir, f'rel_index_{task_i}.json')
        page_file = os.path.join(dataset.data_dir, f'pages_{task_i}.json')
        gist_file = os.path.join(dataset.data_dir, f'gist_{task_i}.json')
        log_file = os.path.join(dataset.data_dir, f'generation_wo_c_{r_tool}_{reasoning_style}_{task_i}.jsonl')
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

        