from .base import *

# General pipeline prompts
class GeneralPrompt:
    
    @staticmethod
    def pagination(preceding:str, passage:str, end_tag:str):
        """
        You are given a passage that is taken from a larger text (article, book, ...) and some numbered labels between the paragraphs in the passage.
        Numbered label are in angeled brackets. For example, if the label number is 19, it shows as <19> in text.
        Please choose one label that it is natural to break reading.
        Such point can be scene transition, end of a dialogue, end of an argument, narrative transition, etc.
        Please answer the break point label and explain.
        For example, if <57> is a good point to break, answer with \"Break point: <57>\n Because ...\"

        Passage:
        {preceding}
        {passage}
        {end_tag}
        """
        return f"""\nYou are given a passage that is taken from a larger text (article, book, ...) and some numbered labels between the paragraphs in the passage.\nNumbered label are in angeled brackets. For example, if the label number is 19, it shows as <19> in text.\nPlease choose one label that it is natural to break reading.\nSuch point can be scene transition, end of a dialogue, end of an argument, narrative transition, etc.\nPlease answer the break point label and explain.\nFor example, if <57> is a good point to break, answer with \"Break point: <57>\n Because ...\"\n\nPassage:\n{preceding}\n{passage}\n{end_tag}\n"""
    
    @staticmethod
    def shorten(passage:str):
        """
        Please shorten the following passage.
        Just give me a shortened version. DO NOT explain your reason.

        Passage:
        {passage}
        """
        return f"""\nPlease shorten the following passage.\nJust give me a shortened version. DO NOT explain your reason.\n\nPassage:\n{passage}\n"""

    @staticmethod
    def answer(question_type:str, article:str, question:str, answer_format:str):
        '''
        Read the following article and answer a {question_type}.

        Article:
        {article}

        Question:
        {question}

        {answer_format}
        '''
        return f'''\nRead the following article and answer a {question_type}.\n\nArticle:\n{article}\n\nQuestion:\n{question}\n\n{answer_format}\n'''
    

# Reading Agent prompts
class ReadingAgentPrompt(GeneralPrompt):
    
    @staticmethod
    def parallel_lookup(article, question):
        """
        The following text is what you remembered from reading an article and a question related to it.
        You may read 5 page(s) of the article again to refresh your memory to prepare yourselve for the question.
        Please respond with which page(s) you would like to read.
        For example, if your only need to read Page 8, respond with \"I want to look up Page [8] to ...\";
        if your would like to read Page 7 and 12, respond with \"I want to look up Page [7, 12] to ...\";
        if your would like to read Page 2, 3, 7, 15 and 18, respond with \"I want to look up Page [2, 3, 7, 15, 18] to ...\".
        if your would like to read Page 3, 4, 5, 12, 13 and 16, respond with \"I want to look up Page [3, 3, 4, 12, 13, 16] to ...\".
        DO NOT answer the question yet.

        Text:
        {article}

        Question:
        {question}

        Take a deep breath and tell me: Which 5 page(s) would you like to read again?
        """
        return f"""\nThe following text is what you remembered from reading an article and a question related to it.\nYou may read 5 page(s) of the article again to refresh your memory to prepare yourselve for the question.\nPlease respond with which page(s) you would like to read.\nFor example, if your only need to read Page 8, respond with \"I want to look up Page [8] to ...\";\nif your would like to read Page 7 and 12, respond with \"I want to look up Page [7, 12] to ...\";\nif your would like to read Page 2, 3, 7, 15 and 18, respond with \"I want to look up Page [2, 3, 7, 15, 18] to ...\".\nif your would like to read Page 3, 4, 5, 12, 13 and 16, respond with \"I want to look up Page [3, 3, 4, 12, 13, 16] to ...\".\nDO NOT answer the question yet.\n\nText:\n{article}\n\nQuestion:\n{question}\n\nTake a deep breath and tell me: Which 5 page(s) would you like to read again?\n"""
    
    @staticmethod
    def sequential_lookup(article, question, retrieved):
        """
        The following text is what you remember from reading a meeting transcript, followed by a question about the transcript.
        You may read multiple pages of the transcript again to refresh your memory and prepare to answer the question.

        Text:
        {article}

        Question:
        {question}

        Please specify a SINGLE page you would like to read again to gain more information or say "STOP" if the information is adequate to answer the question.
        For example, if your would like to read Page 7, respond with \"I want to look up Page 7 to ...\";
        if your would like to read Page 13, respond with \"I want to look up Page 13 to ...\";
        if you think the information is adequate, simply respond with \"STOP\".
        Pages {retrieved} have been looked up already and DO NOT ask to read them again.
        DO NOT answer the question in your response.
        """
        return f"""\nThe following text is what you remember from reading a meeting transcript, followed by a question about the transcript.\nYou may read multiple pages of the transcript again to refresh your memory and prepare to answer the question.\n\nText:\n{article}\n\nQuestion:\n{question}\n\nPlease specify a SINGLE page you would like to read again to gain more information or say "STOP" if the information is adequate to answer the question.\nFor example, if your would like to read Page 7, respond with \"I want to look up Page 7 to ...\";\nif your would like to read Page 13, respond with \"I want to look up Page 13 to ...\";\nif you think the information is adequate, simply respond with \"STOP\".\nPages {retrieved} have been looked up already and DO NOT ask to read them again.\nDO NOT answer the question in your response.\n"""
    


# LongDoc prompt
class LongDocPrompt(GeneralPrompt):
    
    @staticmethod
    def remove_multi_ws(text:str):
        return re.sub('    +', '', text).strip('\n')
    
    class ListEnt:
    
        def _format(self):
            return LongDocPrompt.remove_multi_ws('''
                Generate your response in the following format:
                Important entities:
                1. Entity 1
                2. Entity 2
                3. Entity 3
                ...
            ''')
        
        def match_entities(self, target_ents:List[str], refer_ents:List[str], target_ents_emb:np.ndarray, refer_ents_emb:np.ndarray, top_k:int=1, retrieval_guaranteed:bool=False, simlarity_threhold:float=0.8):
            sim_mat:np.ndarray = np.matmul(target_ents_emb, refer_ents_emb.T)
            ent_map:Dict[str, List[str]] = defaultdict(list)
            for eid, ent in enumerate(target_ents):
                for idx in np.argsort(sim_mat[eid])[::-1][:top_k]:
                    if sim_mat[eid, idx] > simlarity_threhold:
                        ent_map[ent].append(refer_ents[idx])
                    else:
                        if retrieval_guaranteed:
                            ent_map[ent].append(refer_ents[idx])
                        break
            return ent_map
        
        def parse(self, responses:List[str], encoder:Callable, simlarity_threhold:float):
            ent_lists:List[str] = []
            ent_cnt = Counter()
            ent_cnt_threshold = len(responses) // 2 + 1
            for response in responses:
                i = 1
                temp_ents:List[str] = []
                for line in response.splitlines():
                    if line.startswith(f'{i}. '):
                        temp_ents.append(line.split(' ', 1)[1].strip().strip('.'))
                        i += 1
                temp_ents = [s[0].upper() + s[1:] for s in temp_ents]
                ent_lists.append(temp_ents)
                ent_cnt.update(temp_ents)
            g = nx.Graph()
            for list1, list2 in itertools.combinations(ent_lists, 2):
                target_ents_emb:np.ndarray = encoder(list1)
                refer_ents_emb:np.ndarray = encoder(list2)
                g.add_edges_from([(e1, e2[0]) for e1, e2 in self.match_entities(list1, list2, target_ents_emb, refer_ents_emb, simlarity_threhold=simlarity_threhold).items()])
            ent_cluster:Set[str]
            rep_cnt:Dict[str, int] = {}
            for ent_cluster in nx.connected_components(g):
                cnts = [(ent_cnt[ent], ent) for ent in ent_cluster]
                cnts.sort(key=lambda x: x[0], reverse=True)
                rep_cnt[cnts[0][1]] = sum([cnt for cnt, _ in cnts])
            return [rep for rep, cnt in rep_cnt.items() if cnt >= ent_cnt_threshold]
        
        def __call__(self, passage:str):
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}

                List the important named entities in the above passage that are relevant to most of its content.
                Don't give any explanation.
                
                {self._format()}
            ''')
            
    class EntDescription:
        
        def _format(self, important_ents_0:str, important_ents_1:str=None):
            return LongDocPrompt.remove_multi_ws(f'''
                Generate your response using the following format:
                {important_ents_0}: the information of {important_ents_0}
                {important_ents_1}: the information of {important_ents_1}
                ...
                ''' if important_ents_1 is not None else f'''
                Generate your response using the following format:
                {important_ents_0}: the information of {important_ents_0}
            ''')
    
        def parse(self, response:str, important_ents:List[str]):
            description_dict:Dict[str, str] = {}
            for line in response.splitlines():
                if ': ' in line:
                    ent, description = line.split(': ', 1)
                    ent = ent.strip()
                    if '.' in ent and ent[:ent.index('.')].isnumeric():
                        ent = ent.split('.', 1)[1].strip()
                    ent = ent.strip('+* ')
                    description = description.strip()
                    if ent in important_ents:
                        description_dict[ent] = description
            return description_dict
        
        def __call__(self, passage, important_ents:List[str]):
            important_ents_str = '\n'.join(important_ents)
            important_ents_0 = important_ents[0]
            important_ents_1 = important_ents[1] if len(important_ents) > 1 else None
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}

                Based on the above passage, briefly and truthfully describe the information of each following entity:
                {important_ents_str}

                {self._format(important_ents_0, important_ents_1)}
            ''')

    class RelDescription:
        
        def _format(self):
            return LongDocPrompt.remove_multi_ws('''
                Generate your response in the following format:
                1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.
                2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.
                ...
            ''')
    
        def parse(self, response:str, ents:List[str]):
            relations:List[Tuple[List[str], str]] = []
            for line in response.splitlines():
                if ':' in line:
                    if '.' in line and line[:line.index('.')].isnumeric():
                        line = line.split('.', 1)[1].strip()
                    line = line.strip('+* ')
                    relation = line.split(':', 1)[1]
                    relation = relation.strip()
                    relation_description = line.lower()
                    related_ents = [ent for ent in ents if ent.lower() in relation_description]
                    if len(related_ents) > 1:
                        relations.append((related_ents, relation))
            return relations
        
        def __call__(self, passage, important_ents:List[str]):
            important_ents_str = '\n'.join(important_ents)
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}

                Important entities:
                {important_ents_str}

                Above is a passage and the important entities in the passage.
                Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above passage.
                Try to include as many entities as possible in each cluster.
                
                {self._format()}
            ''')

    class PairwiseRelDescription:
        def _format(self):
            return LongDocPrompt.remove_multi_ws('''
                Suppose Entity 1 and 3 are from entity set one and Entity 2 and 4 are from entity set two.
                Your response should be in the following format:
                1. (Entity 1, Entity 2): summary of relational information between Entity 1 and 2.
                2. (Entity 3, Entity 4): summary of relational information between Entity 3 and 4.
                ...
            ''')
    
        def parse(self, response:str, ent1s:List[str], ent2s:List[str]):
            lowered_ent1s, lowered_ent2s = [e.lower() for e in ent1s], [e.lower() for e in ent2s]
            relations:List[Tuple[List[str], str]] = []
            for line in response.splitlines():
                line = line.strip()
                if ':' in line:
                    if '.' in line and line[:line.index('.')].isnumeric():
                        line = line.split('.', 1)[1].strip()
                    line = line.strip('+* ')
                    ents, relation = line.split(':', 1)
                    relation = relation.strip()
                    if ents.count(',') != 1:
                        continue
                    ents = ents.strip()
                    if ents.startswith('(') and ents.endswith(')'):
                        ents = ents[1:-1].strip()
                    ent1, ent2 = ents.split(',')
                    ent1, ent2 = ent1.strip(), ent2.strip()
                    if any([lowered_ent1.startswith(ent1.lower()) for lowered_ent1 in lowered_ent1s]) and any([lowered_ent2.startswith(ent2.lower()) for lowered_ent2 in lowered_ent2s]):
                        relations.append(([ent1, ent2], relation))
            return relations

        def __call__(self, passage, important_ent1s:List[str], important_ent2s:List[str]):
            important_ent1s_str, important_ent2s_str = ', '.join(important_ent1s), ', '.join(important_ent2s)
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}

                Important entity set one:
                {important_ent1s_str}
                
                Important entity set two:
                {important_ent2s_str}

                Above is a passage and two sets of important entities in the passage.
                Find the related entity pairs, where each pair consists of an entity from each important entity set, and use one sentence to informatively summarize their relational information in the above passage.
                
                {self._format()}
            ''')
        
    class ChunkRewrite:
        def _format(self, chunk_ids:List[int]):
            formats = '\n\n'.join([f'''The summary of chunk {chunk_id + 1}:\n[Summary]''' for chunk_id in chunk_ids])
            return LongDocPrompt.remove_multi_ws(f'''
                Generate your response in the following format:
                {formats}
            ''')
    
        def parse(self, response:str):
            summary_header = 'The summary of chunk '
            header_len = len(summary_header)
            
            summaries = [summary[summary.lower().index(summary_header.lower()):].split(':', 1) for summary in response.split('\n\n') if summary_header.lower() in summary.lower()]
            summaries = [summary for summary in summaries if summary[0][header_len:].isnumeric()]
            return {int(summary[0][header_len:]) - 1 : summary[1].strip() for summary in summaries if summary[1].strip()}
    
        def __call__(self, pages:List[str], chunk_ids:List[int]):
            passage = '\n\n'.join([f'Chunk {sid + 1}: {sent}' for sid, sent in enumerate(pages)])
            if len(chunk_ids) == 1:
                chunk_ids = set(chunk_ids)
                chunk_ids.update(np.random.choice(len(pages), 2))
                chunk_ids = list(chunk_ids)
                chunk_ids.sort()
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}

                Above is a passage splitted into {len(pages)} chunks.
                Please summarize the chunks {[cid + 1 for cid in chunk_ids]} separately so that each summarized chunk can be understood independently. 
                You should make use of the general context to correctly understand each chunk.
                Each summarized chunk should be a sequence of statements in thrid-person narration, briefly describe the information in the chunk, with all the coreferences resolved.

                {self._format(chunk_ids)}
            ''')
    
    class Compare:
        def _format(self):
            return LongDocPrompt.remove_multi_ws('''
                Generate your response in the following format:
                Commonality:
                Both the "Current Passage" and the "Passage After" mention ...
                
                Distinction:
                Unique in "Current Passage":
                The "Current Passage" uniquely mentions ...
                
                Unique in "Passage After":
                The "Passage After" uniquely mentions ...
            ''')
        
        def parse(self, response:str):
            result:Dict[str, str] = {}
            for line in response.split('\n\n'):
                line = line.strip()
                if line.startswith('Commonality:'):
                    result['com'] = line.split(':', 1)[1].strip()
                else:
                    if 'Unique in "Current Passage":' in line:
                        result['uic'] = line[line.index('Unique in "Current Passage":'):].split(':', 1)[1].strip()
                    elif 'Unique in "Passage After":' in line:
                        result['uia'] = line[line.index('Unique in "Passage After":'):].split(':', 1)[1].strip()
            return result
        
        def __call__(self, current_passage:str, passage_after:str):
            return LongDocPrompt.remove_multi_ws(f'''
                Current Passage:
                {current_passage}
                
                Passage After:
                {passage_after}
                
                Above are two consecutive passages from a document, namely "Current Passage" and "Passage After".
                Please compare the shared commonalities and unshared distinctions between the "Current Passage" and the "Passage After".
                You should make use of the general context to correctly understand each passage.
                The commonality should briefly describe the information in common between two passages.
                The distinction should briefly describe the information unique in each passage.
                All the response should be a sequence of statements in thrid-person narration with all the coreferences resolved.
                
                {self._format()}
            ''')
    
    class Summary:
        def _format(self):
            return LongDocPrompt.remove_multi_ws('''
                Generate your response in the following format:
                Shortened paragraph: [The shortened paragraph of the passage ...]
            ''')
    
        def parse(self, response:str):
            header_str = 'Shortened paragraph:'
            lowered_header_str = header_str.lower()
            response = ' '.join(response.split())
            lowered_response = response.lower()
            if lowered_header_str in lowered_response:
                start_idx = lowered_response.index(lowered_header_str) + len(header_str)
                return response[start_idx:].strip()
            else:
                return response
                
        def __call__(self, passage:str | List[str]):
            if isinstance(passage, list):
                passage = ' '.join(passage)
            
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}
                
                Please shorten the passage above into a 100-word paragraph.
                Your response should be a sequence of fluent statements in thrid-person narration.
                
                {self._format()}
            ''')
        
    class SummaryEval:
        def _format(self):
            return LongDocPrompt.remove_multi_ws('''
                Return "Yes" if all the information in the summary is correct and "No" if there is any incorrect information.
                Do not explain your answer.
            ''')

        def parse(self, response:str):
            return 'yes' in response.lower() and 'no' not in response.lower()
    
        def __call__(self, passage:str, summary:str):
            return LongDocPrompt.remove_multi_ws(f'''
                Passage:
                {passage}
                
                Summary:
                {summary}
                
                Above is a passage and its summary.
                Does the summary correctly summarize the passage?
                
                {self._format()}
            ''')
            
    