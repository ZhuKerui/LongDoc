from .data import *

def concate_with_overlap(pieces:List[str], chunk_size:int=10, overlap:int=1):
    chunk_size_with_overlap = chunk_size + overlap
    return [pieces[batch_start * chunk_size : batch_start * chunk_size + chunk_size_with_overlap] for batch_start in range((len(pieces) + 1) // chunk_size)]




summary_prompt = '''Summarize the following passage.

Passage:
{chunk}'''

statement_prompt = '''A summary of a story and a passage from the story is provided below.


Summary:
{summary}


Passage:
{chunk}


A summary of a story and a passage from the story is provided above.
Break the passage into a comprehensive list of atomic fact statements.
The statements should be ordered as they appear in the passage.
Try to use the original words from the passage.
You should use the summary as the context to help you better understand the passage.'''


import spacy
from rank_bm25 import BM25Okapi
import spacy.tokens

class ChunkInfo(BaseModel):
    i: int
    chunk_text: str
    statements: List[str] = []
    entities: List[List[str]] = []
    keywords: List[List[str]] = []
        

class LongDoc:
    
    def __init__(self, factory:Factory, chunk_info_file:str=None) -> None:
        self.factory = factory
        self.nlp = spacy.load('en_core_web_lg')
        if chunk_info_file:
            self.chunk_infos = [ChunkInfo.parse_obj(line) for line in read_json(chunk_info_file)]
            self.build_relation_graph()
            self.build_lexical_store()
        
    def build_index(self, article:str, chunk_info_file:str=None):
        pieces = self.factory.split_text(article)
        print('generate_statements')
        chunks, statements = self.generate_statements(pieces)
        # statements = [ci.statements for ci in chunk_infos]
        print('end')
        chunk_infos = [ChunkInfo(i=i, chunk_text=chunk) for i, chunk in enumerate(chunks)]
        missing_chunk_ids = [ci.i for ci in chunk_infos if not ci.statements]
        attempt = 0
        while missing_chunk_ids:
            print(attempt, len(missing_chunk_ids))
            attempt += 1
            if len(missing_chunk_ids) == len(statements):
                entities = self.extract_entities(statements)
                for cid, (statement_list, entity_list) in enumerate(zip(statements, entities)):
                    if len(statement_list) == len(entity_list):
                        chunk_infos[cid].statements, chunk_infos[cid].entities = statement_list, entity_list
            else:
                temp_stm_groups = []
                for missing_ci in missing_chunk_ids:
                    temp_stms = statements[missing_ci]
                    split = np.random.randint(len(statements) // 3, len(statements) * 2 // 3)
                    temp_stm_groups.append(temp_stms[:split])
                    temp_stm_groups.append(temp_stms[split:])
                entities = self.extract_entities(temp_stm_groups)
                for sid in range(len(temp_stm_groups)//2):
                    if len(temp_stm_groups[sid * 2]) == len(entities[sid * 2]) and len(temp_stm_groups[sid * 2 + 1]) == len(entities[sid * 2 + 1]):
                        chunk_infos[missing_chunk_ids[sid]].statements, chunk_infos[missing_chunk_ids[sid]].entities = temp_stm_groups[sid * 2] + temp_stm_groups[sid * 2 + 1], entities[sid * 2] + entities[sid * 2 + 1]
            missing_chunk_ids = [ci.i for ci in chunk_infos if not ci.statements]

        for ci in chunk_infos:
            for sid, statement in enumerate(ci.statements):
                addition_ents, keywords = self.collect_keywords_from_text(statement)
                ci.keywords.append(keywords)
                for addition_ent in addition_ents:
                    if all([addition_ent.lower() not in ent.lower() for ent in ci.entities[sid]]):
                        ci.entities[sid].append(addition_ent)
        
        self.chunk_infos = chunk_infos
        self.build_relation_graph()
        self.build_lexical_store()
        if chunk_info_file:
            write_json(chunk_info_file, [ci.dict() for ci in chunk_infos])

    def collect_keywords_from_text(self, text:str):
        
        def trim_det_adv(noun_chunk:spacy.tokens.Span):
            for tid, t in enumerate(noun_chunk):
                if t.pos_ not in ['DET', 'ADV']:
                    return noun_chunk[tid:]
                
        doc = self.nlp(text)
        ncs = [trim_det_adv(nc) for nc in doc.noun_chunks if nc.root.pos_ not in ['NUM', 'PRON']]
        ents = [trim_det_adv(ent) for ent in doc.ents if ent.root.pos_ not in ['NUM', 'PRON']]
        keywords = [(t.i, t.i+1) for t in doc if t.pos_ in ['VERB', 'ADJ', 'ADV']]
        ncs_spans = [(nc.start, nc.end) for nc in ncs if nc]
        ents_spans = [(ent.start, ent.end) for ent in ents if ent]
        nc_id, eid = 0, 0
        spans = []
        while nc_id < len(ncs_spans) and eid < len(ents_spans):
            nc_span, ent_span = ncs_spans[nc_id], ents_spans[eid]
            if set(range(*nc_span)).intersection(range(*ent_span)):
                merged_span = (min(nc_span[0], ent_span[0]), max(nc_span[1], ent_span[1]))
                spans.append(merged_span)
                nc_id += 1
                eid += 1
            else:
                if nc_span[0] < ent_span[0]:
                    spans.append(nc_span)
                    nc_id += 1
                else:
                    spans.append(ent_span)
                    eid += 1
        spans.extend(ncs_spans[nc_id:])
        spans.extend(ents_spans[eid:])
        updated_spans:List[Tuple[int, int]] = []
        for span in spans:
            doc_span = doc[span[0]:span[1]]
            if ',' in doc_span.text:
                start = doc_span.start
                for t in doc_span:
                    if t.text == ',':
                        if t.i != start:
                            updated_spans.append((start, t.i))
                        start = t.i + 1
                if start < span[1]:
                    updated_spans.append((start, span[1]))
            else:
                updated_spans.append(span)
        updated_spans = [span for span in updated_spans if any([t.pos_ in ['NOUN', 'PROPN'] for t in doc[span[0]:span[1]]])]
        updated_spans = [span if doc[span[0]].pos_ != 'PRON' else (span[0]+1, span[1]) for span in updated_spans]
        ent_candidates = [doc[span[0]:span[1]].text for span in updated_spans]
        ent_candidates = [ent.strip('"') for ent in ent_candidates]
        ent_candidates = [ent for ent in ent_candidates if len(ent) >= 2]
        
        keywords = updated_spans + keywords
        keywords.sort(key=lambda x: x[0])
        kw_candidates = [doc[span[0]:span[1]].text for span in keywords]
        kw_candidates = [ent.strip('"') for ent in kw_candidates]
        return ent_candidates, kw_candidates
    
    def parse_statements(self, text:str):
        i = 1
        statements:List[str] = []
        for line in text.strip().splitlines():
            if line.startswith(f'{i}. '):
                statements.append(line.split(' ', 1)[1].strip())
                i += 1
        return statements

    def parse_entities(self, text:str):
        i = 1
        list_of_ent_list:List[List[str]] = []
        for line in text.strip().splitlines():
            line = line.strip()
            if line.startswith(f'{i}. '):
                temp_ent_list = line.split(' ', 1)[1].strip().split(',')
                ent_list:List[str] = []
                incomplete_ent = []
                for ent in temp_ent_list:
                    if '(' in ent and ')' not in ent:
                        incomplete_ent.append(ent)
                    elif '(' not in ent and ')' in ent:
                        incomplete_ent.append(ent)
                        ent_list.append(','.join(incomplete_ent).strip().strip('.'))
                        incomplete_ent.clear()
                    elif incomplete_ent:
                        incomplete_ent.append(ent)
                    else:
                        ent_list.append(ent.strip().strip('.'))
                ent_list = [ent.split(':', 1)[1].strip() if ent.startswith('Entities:') else ent for ent in ent_list]
                ent_list = [self.clean_entity(ent) for ent in ent_list]
                ent_list = [ent for ent in ent_list if ent]
                list_of_ent_list.append(ent_list)
                i += 1
        return list_of_ent_list
    
    def build_relation_graph(self):
        relation_graph = nx.Graph()
        # semantic edges
        for ci in self.chunk_infos:
            for sid, related_ents in enumerate(ci.entities):
                loc = (ci.i, sid)
                # Insert node locs
                for e in related_ents:
                    if not relation_graph.has_node(e):
                        relation_graph.add_node(e, locs=[], norm=' '.join(self.normalize_entity(e)))
                    ent_locs:list = relation_graph.nodes[e]['locs']
                    if loc not in ent_locs:
                        ent_locs.insert(0, loc)
                for ent1, ent2 in itertools.combinations(related_ents, 2):
                    if not relation_graph.has_edge(ent1, ent2):
                        relation_graph.add_edge(ent1, ent2, locs=[])
                    edge_locs:list = relation_graph[ent1][ent2]['locs']
                    edge_locs.append((loc, loc))
        
        self.normal2ents:Dict[str, List[str]] = defaultdict(list)
        for ent, normal in relation_graph.nodes(data='norm'):
            self.normal2ents[normal].append(ent)
        normals = list(self.normal2ents)
        normals.sort()
        self.ent_corpus = [ent.split() for ent in normals]
        self.ent_bm25 = BM25Okapi(self.ent_corpus)
        
        for normal in self.normal2ents:
            # Add edges between entities that have the same norm
            for ent1, ent2 in itertools.combinations(self.normal2ents[normal], 2):
                if not relation_graph.has_edge(ent1, ent2):
                    relation_graph.add_edge(ent1, ent2, locs=[])
                edge_locs:list = relation_graph[ent1][ent2]['locs']
                edge_locs.extend(itertools.product(relation_graph.nodes[ent1]['locs'], relation_graph.nodes[ent2]['locs']))
            # Add edges between entities that have similar norms
            scores:List[float] = self.ent_bm25.get_scores(normal.split()).tolist()
            for score, similar_normal in zip(scores, self.ent_corpus):
                if score > 0:
                    similar_normal = ' '.join(similar_normal)
                    if normal != similar_normal:
                        for ent1, ent2 in itertools.product(self.normal2ents[normal], self.normal2ents[similar_normal]):
                            if not relation_graph.has_edge(ent1, ent2):
                                relation_graph.add_edge(ent1, ent2, locs=[])
                            edge_locs:list = relation_graph[ent1][ent2]['locs']
                            edge_locs.extend(itertools.product(relation_graph.nodes[ent1]['locs'], relation_graph.nodes[ent2]['locs']))
        
        self.graph = relation_graph
    
    def build_lexical_store(self):
        self.raw_corpus = [self.normalize_text(ci.chunk_text) for ci in self.chunk_infos]
        self.raw_bm25 = BM25Okapi(self.raw_corpus)
        
    def lexical_retrieval_chunks(self, query:str, n:int=5):
        chunk_idxs = self.bm25_retrieve(self.normalize_text(query), self.raw_bm25)
        return [(self.chunk_infos[idx].chunk_text, idx) for idx in chunk_idxs][:n]
    
    def lexical_retrieval_entities(self, query:str, n:int=5):
        tokenized_query = self.normalize_entity(query)
        normal_idxs = self.bm25_retrieve(tokenized_query, self.ent_bm25)
        candidate_normals = [' '.join(self.ent_corpus[idx]) for idx in normal_idxs]
        temp_ents = []
        for normal in candidate_normals:
            temp_ents.extend(self.normal2ents[normal])
        temp_ent_corpus = [self.split_lower_text(ent) for ent in temp_ents]
        temp_bm25 = BM25Okapi(temp_ent_corpus)
        ent_idxs = self.bm25_retrieve(tokenized_query, temp_bm25)
        return [temp_ents[idx] for idx in ent_idxs][:n]
    
    def exact_match_chunks(self, query:str):
        normalized_query = ' '.join(self.normalize_text(query))
        return [(self.chunk_infos[idx].chunk_text, idx) for idx, normalized_chunk in enumerate(self.raw_corpus) if normalized_query in ' '.join(normalized_chunk)]
        
    def generate_statements(self, pieces:List[str], chunk_size:int=5, summary_size:int=25, overlap:int=1):
        summary_chunks = concate_with_overlap(pieces, summary_size, overlap=overlap)
        summaries = self.factory.llm.generate([[HumanMessage(content=summary_prompt.format(chunk=' '.join(summary_chunk)))] for summary_chunk in summary_chunks])
        prompts = []
        chunks:List[str] = []
        for summary_chunk, summary in zip(summary_chunks, summaries.generations):
            temp_summary_size = min(summary_size, len(summary_chunk))
            summary_chunk = summary_chunk[:temp_summary_size]
            for batch_start in range((temp_summary_size + 1) // chunk_size):
                chunk = ' '.join(summary_chunk[batch_start * chunk_size : (batch_start + 1) * chunk_size])
                prompts.append(statement_prompt.format(summary=summary[0].text, chunk=chunk))
                chunks.append(chunk)
        return chunks, [self.parse_statements(gen[0].text) for gen in self.factory.llm.generate([[HumanMessage(content=prompt)] for prompt in prompts]).generations]
        
    def clean_entity(self, ent_text:str):
        ent_doc = self.nlp(ent_text, disable=['parser', 'ner'])
        for tid, t in enumerate(ent_doc):
            if t.pos_ not in ['DET', 'CCONJ', 'PRON']:
                return ent_doc[tid:].text
            
    def normalize_entity(self, ent_text:str):
        # return [t.text.lower() if t.pos_ != 'NOUN' else t.lemma_.lower() for t in self.nlp(ent_text, disable=['parser', 'ner']) if t.pos_ not in ['DET', 'PUNCT', 'ADP', 'SCONJ', 'PRON', 'CCONJ', 'PART', 'AUX']]
        return [t.text.lower() if t.pos_ != 'NOUN' else t.lemma_.lower() for t in self.nlp(ent_text, disable=['parser', 'ner']) if not (t.is_stop or t.pos_ == "PUNCT")]
    
    def normalize_text(self, text:str):
        return [t.lemma_.lower() if t.pos_ in ['NOUN', 'VERB'] else t.text.lower() for t in self.nlp(text, disable=['ner', 'parser']) if not t.is_stop]
    
    def split_lower_text(self, text:str) -> List[str]:
        return [t.text.lower() for t in self.nlp(text, disable=['ner', 'parser'])]
    
    def bm25_retrieve(self, tokenized_query:List[str], bm25:BM25Okapi):
        index_score_pairs = [(idx, score) for idx, score in enumerate(bm25.get_scores(tokenized_query)) if score > 0]
        index_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in index_score_pairs]
        
    def extract_entities(self, statements:List[List[str]]):
        return [self.parse_entities(gen[0].text) for gen in self.factory.llm.generate([[HumanMessage(content='List the entities in each line of the following statements.\n\nStatements:\n' + '\n'.join([f'{sid+1}. {s}' for sid, s in enumerate(statement)]))] for statement in statements]).generations]

if __name__ == '__main__':
    
    import argparse
    import os
    from pathlib import Path
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--task', type=str, default='quality', choices=['narrativeqa', 'quality', 'musique'])
    # parser.add_argument('--r_tool', type=str, default='index', choices=['dpr', 'index', 'gist'])
    # parser.add_argument('--reasoning_style', type=str, default='p', choices=['s', 'p'])
    # args = parser.parse_args()

    # context_type = 'novel'

    # task_name = args.task
    # r_tool = args.r_tool
    # reasoning_style = args.reasoning_style
    
    # llm = LLM()
    # retriever = Retriever()
    # # llm = 'mistralai/Mistral-7B-Instruct-v0.2'
    
    # if task_name == 'narrativeqa':
    #     dataset = NarrativeQADataset(llm)
    # elif task_name == 'musique':
    #     dataset = MuSiQueDataset(llm)
    # else:
    #     dataset = QualityDataset(llm, 'dev')
        
    # longdoc = LongDoc(dataset, retriever, llm)
    # reading_agent = ReadingAgent(dataset, llm)
    
    # for task_i in range(0, 10):
    #     print(f'{task_i} start')
        
    #     index_file = os.path.join(dataset.data_dir, f'rel_index_{task_i}.json')
    #     page_file = os.path.join(dataset.data_dir, f'pages_{task_i}.json')
    #     gist_file = os.path.join(dataset.data_dir, f'gist_{task_i}.json')
    #     log_file = os.path.join(dataset.data_dir, f'generation_wo_c_{r_tool}_{reasoning_style}_{task_i}.jsonl')
    #     questions, _ = dataset.get_questions_and_answers(dataset.data[task_i])
    #     for qid, query in enumerate(questions):
    #         if r_tool == 'gist':
    #             reading_agent.main(query, page_file, gist_file, log_file, reasoning_style)
    #         else:
    #             if reasoning_style == 'p':
    #                 longdoc.main(query, index_file, log_file, r_tool)
    #             else:
    #                 longdoc.self_rag(query, page_file, log_file)
            
    #     print(f'{task_i} end')

        