from rouge_metric import PyRouge
from .data import Dataset
from .models import Retriever, LLM, LLMServer
from .base_utils import *
from .prompt import *
import itertools
from nltk import word_tokenize

def cal_rouge(hp, ref):
    return PyRouge().evaluate([hp], [[ref]])

class LongDoc:
    
    def __init__(self, dataset:Dataset, retriever:Retriever, llm:str | LLM="mistralai/Mistral-7B-Instruct-v0.2") -> None:
        self.dataset = dataset
        if llm:
            self.llm_server = LLMServer(llm)
        self.retriever = retriever

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
            chat_response = self.llm_server(list_entity_prompt)[0]
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
        return DocIndex(all_graph, paragraphs, all_summary, self.retriever.embed_paragraphs(paragraphs), pid2nodes)
    
    def identify_noun_verb(self, query:str, n:int=10):
        query_entity_prompt = f'''Question: {query}\nYou need to answer the above question based on a given story. Before reading the story, identify important and unique noun and verb phrases in the question that you want to query from the story for useful information. All the phrases must appear in the question. Don't give any explanation. Generate your response in the following format:\n"Query noun phrases:\nthe first noun phrase, the second noun phrase, ...\n\nQuery verb phrases:\nthe first verb phrase, the second verb phrase, ...".'''
        chat_response = self.llm_server(query_entity_prompt, n=n, temperature=0.8)
        noun_phrases = Counter()
        verb_phrases = Counter()
        is_noun = False
        is_verb = False
        for choice in chat_response:
            for line in choice.splitlines():
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
        node_embeds = self.retriever.embed_paragraphs(nodes, normalize=True)
        for target in targets:
            ent_embed = self.retriever.embed_paragraphs([target], normalize=True)
            scores:np.ndarray = node_embeds.dot(ent_embed.squeeze())
            # if scores.max() < threshold:
            #     self.add_node(doc_index, target)
            #     if doc_index.graph.has_node(target):
            #         ret.append([target])
            #         nodes = list(doc_index.graph.nodes)
            #         nodes.sort()
            #         node_embeds = self.retriever.embed_paragraphs(nodes, normalize=True)
            #         continue
            max_indices = np.argsort(scores)[::-1][:k]
            ret.append([nodes[i] for i in max_indices if scores[i] >= threshold] + [target])
        return ret

    def add_node(self, doc_index:DocIndex, node:str):
        pids = self.retriever.dense_retrieval(node, doc_index.paragraph_embs, 10)
        for pid in pids:
            important_ents = "\n".join(doc_index.pid2nodes[pid])
            response = self.llm_server(f'''Context:\n{doc_index.paragraphs[pid]}\n\nImportant entities:\n{important_ents}\n\nDoes the above context contain information about "{node}"? If yes, summarize the information in the above context for "{node}" and try to include other important entities in the summary if they are related. If no, simply reply "No". Generate your response in the following format: "Answer: summary of {node} or 'No.'"''')[0]
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

            retrieval_command = self.llm_server(passage_retrieve_prompt)[0]
            log_info(log_file, 'retrieval_command', retrieval_command)
            pids = self.retrieval_passage(retrieval_command, doc_index)
            assert pids
            
        elif r_tool == 'dpr':
            pids = self.retriever.dense_retrieval(query, doc_index.paragraphs, 5)
            pids.sort()
        
        retrieval_result = ''.join([f'''Passage {pid}:\n{doc_index.paragraphs[pid]}\n\n''' for pid in pids])
        log_info(log_file, 'retrieval_result', retrieval_result)
        
        # Step 3: analyze retrieved info
        generation = self.llm_server(GeneralPrompt.answer(self.dataset.question_type, retrieval_result, query, self.dataset.answer_format))[0]
        log_info(log_file, 'generation', generation)
        

    def self_rag(self, init_question:str, page_file:str, log_file:str=None):
        paragraphs = ['\n'.join(page) for page in read_json(page_file)]
        state = 'retrieve'
        context = ''
        generation = ''
        retrieved_pids = []
        log_info(log_file, 'query', init_question)
        init_query = init_question.split('\n')[0]
        query = init_query
        repeat = 0
        step = 1
        while True:
            if step >= 6:
                generation = self.llm_server(GeneralPrompt.answer(self.dataset.question_type, context, init_question, self.dataset.answer_format))[0]
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
                pids = self.retriever.dense_retrieval(query, paragraphs, 5)
                log_info(log_file, 'candidate_pids', pids)
                new_retrieved_pids = []
                for pid in pids:
                    doc_txt = paragraphs[pid]
                    score_response = self.llm_server(score_template.format(document=doc_txt, question=query))[0]
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
                generation = self.llm_server(prompt_summarize_template.format(article=context, question=init_query), temperature=0.7)[0]
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

                hallucination_response = self.llm_server(hallucination_template.format(documents=context, generation=generation))[0]
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

                    answer_grade_response = self.llm_server(answer_grade_template.format(question=init_query, generation=generation))[0]
                    log_info(log_file, 'answer_grade_response', answer_grade_response)
                    if 'yes' in answer_grade_response.lower():
                        generation = self.llm_server(GeneralPrompt.answer(self.dataset.question_type, context, init_question, self.dataset.answer_format))[0]
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

                re_write_response = self.llm_server(re_write_template.format(question=init_query, query=query, generation=generation))[0]
                query = re_write_response
                state = 'retrieve'
                step += 1
