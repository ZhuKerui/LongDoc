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
    def _ent_list_format():
        '''
        Generate your response in the following format:
        "Important entities:
        1. Entity 1
        2. Entity 2
        3. Entity 3
        ..."
        '''
        return '''Generate your response in the following format:\n"Important entities:\n1. Entity 1\n2. Entity 2\n3. Entity 3\n..."'''
    
    @staticmethod
    def match_entities(target_ents:List[str], refer_ents:List[str], target_ents_emb:np.ndarray, refer_ents_emb:np.ndarray, top_k:int=1, retrieval_guaranteed:bool=False):
        sim_mat:np.ndarray = np.matmul(target_ents_emb, refer_ents_emb.T)
        ent_map:Dict[str, List[str]] = defaultdict(list)
        for eid, ent in enumerate(target_ents):
            for idx in np.argsort(sim_mat[eid])[::-1][:top_k]:
                if sim_mat[eid, idx] > 0.8:
                    ent_map[ent].append(refer_ents[idx])
                else:
                    if retrieval_guaranteed:
                        ent_map[ent].append(refer_ents[idx])
                    break
        return ent_map
    
    @staticmethod
    def parse_entities(responses:List[str], encoder:Callable):
        ent_lists:List[str] = []
        ent_cnt = Counter()
        ent_cnt_threshold = len(responses) // 2 + 1
        for response in responses:
            i = 1
            temp_ents = []
            for line in response.splitlines():
                if line.startswith(f'{i}. '):
                    temp_ents.append(line.split(' ', 1)[1].strip().strip('.'))
                    i += 1
            ent_lists.append(temp_ents)
            ent_cnt.update(temp_ents)
        g = nx.Graph()
        for list1, list2 in itertools.combinations(ent_lists, 2):
            target_ents_emb:np.ndarray = encoder(list1)
            refer_ents_emb:np.ndarray = encoder(list2)
            g.add_edges_from([(e1, e2[0]) for e1, e2 in LongDocPrompt.match_entities(list1, list2, target_ents_emb, refer_ents_emb).items()])
        ent_cluster:Set[str]
        rep_cnt:Dict[str, int] = {}
        for ent_cluster in nx.connected_components(g):
            cnts = [(ent_cnt[ent], ent) for ent in ent_cluster]
            cnts.sort(key=lambda x: x[0], reverse=True)
            rep_cnt[cnts[0][1]] = sum([cnt for cnt, _ in cnts])
        return [rep for rep, cnt in rep_cnt.items() if cnt >= ent_cnt_threshold]
    
    @staticmethod
    def _ent_description_format(important_ents_0:str, important_ents_1:str=None):
        '''
        Generate your response using the following format:
        "{important_ents_0}: the information of {important_ents_0}
        {important_ents_1}: the information of {important_ents_1}
        ..."
        '''
        return f'''Generate your response using the following format:\n"{important_ents_0}: the information of {important_ents_0}\n{important_ents_1}: the information of {important_ents_1}\n..."''' if important_ents_1 is not None else f'''Generate your response using the following format:\n"{important_ents_0}: the information of {important_ents_0}\n..."'''

    @staticmethod
    def parse_ent_description(response:str, important_ents:List[str]):
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

    @staticmethod
    def _relation_description_format():
        '''
        Generate your response in the following format:
        "1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.
        2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.
        ..."
        '''
        return '''Generate your response in the following format:\n"1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.\n2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.\n..."'''
    
    @staticmethod
    def parse_relation_description(response:str, ents:List[str]):
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

    @staticmethod
    def _context_format(passage:str):
        '''
        Passage:
        {passage}
        '''
        return f'''Passage:\n{passage}'''
    
    @staticmethod
    def _context_w_note_format(recap:str, passage:str):
        '''
        Recap:
        {recap}
        
        Current Passage:
        {passage}
        '''
        return f'''Recap:\n{recap}\n\nCurrent Passage:\n{passage}'''
        
    @staticmethod
    def list_entity(passage):
        '''
        {_context_format}

        List the important named entities in the above passage that are relevant to most of its content.
        Don't give any explanation.
        
        {_ent_list_format}
        '''
        return f'''\n{LongDocPrompt._context_format(passage)}\n\nList the important named entities in the above passage that are relevant to most of its content.\nDon't give any explanation.\n\n{LongDocPrompt._ent_list_format()}\n'''

    @staticmethod
    def list_entity_w_note(recap, passage):
        '''
        {_context_w_note_format}

        Above is a recap of several previous passages and the content of the current passage.
        Make use of the recap information to help you better understand the current passage.
        Based on the recap and the current passage, list the important named entities in the current passage that are relevant to most of its content.
        Don't give any explanation.
        
        {_ent_list_format}
        '''
        return f'''\n{LongDocPrompt._context_w_note_format(recap, passage)}\n\nAbove is a recap of several previous passages and the content of the current passage.\nMake use of the recap information to help you better understand the current passage.\nBased on the recap and the current passage, list the important named entities in the current passage that are relevant to most of its content.\nDon't give any explanation.\n\n{LongDocPrompt._ent_list_format()}\n'''

    @staticmethod
    def ent_description(passage, important_ents:List[str]):
        '''
        {_context_format}

        Based on the above passage, briefly and truthfully describe the information of each following entity:
        {important_ents_str}

        {_ent_description_format}
        '''
        important_ents_str = '\n'.join(important_ents)
        return f'''\n{LongDocPrompt._context_format(passage)}\n\nBased on the above passage, briefly and truthfully describe the information of each following entity:\n{important_ents_str}\n\n{LongDocPrompt._ent_description_format(important_ents[0], important_ents[1] if len(important_ents) > 1 else None)}\n'''
    
    @staticmethod
    def ent_description_w_note(recap, passage, important_ents:List[str]):
        '''
        {_context_w_note_format}

        Above is a recap of several previous passages and the current passage.
        Make use of the recap information to help you better understand the current passage.
        You are given a list of entities mentioned in the current passage.
        For each entity:
        1. if it is not mentioned in the recap, briefly and truthfully generate an entity description for it, which should include its basic information and general background in the current passage.
        2. if it is mentioned in the recap, but the entity description is out-dated or faulty, update the entity description with new information in current passage. 
        3. otherwise, skip this entity in your response.
        
        Entities in Current Passage:
        {important_ents_str}

        {_ent_description_format}
        '''
        important_ents_str = '\n'.join(important_ents)
        return f'''\n{LongDocPrompt._context_w_note_format(recap, passage)}\n\nAbove is a recap of several previous passages and the current passage.\nMake use of the recap information to help you better understand the current passage.\nBased on the recap and the current passage, briefly and truthfully describe the information of each following entity in the current passage:\n{important_ents_str}\n\n{LongDocPrompt._ent_description_format(important_ents[0], important_ents[1] if len(important_ents) > 1 else None)}\n'''

    @staticmethod
    def relation_description(passage, important_ents:List[str]):
        '''
        {_context_format}

        Important entities:
        {important_ents_str}

        Above is a passage and the important entities in the passage.
        Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above passage.
        Try to include as many entities as possible in each cluster.
        
        {_relation_description_format}
        '''
        important_ents_str = '\n'.join(important_ents)
        return f'''\n{LongDocPrompt._context_format(passage)}\n\nImportant entities:\n{important_ents_str}\n\nAbove is a passage and the important entities in the passage.\nFind the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above passage.\nTry to include as many entities as possible in each cluster.\n\n{LongDocPrompt._relation_description_format()}\n'''

    @staticmethod
    def relation_description_w_note(recap, passage, important_ents:List[str]):
        '''
        {_context_w_note_format}

        Important entities:
        {important_ents_str}

        Above is a recap of several previous passages, the current passage and the important entities in the current passage.
        Make use of the recap information to help you better understand the current passage.
        Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the current passage.
        Try to include as many entities as possible in each cluster.
        
        {_relation_description_format}
        '''
        important_ents_str = '\n'.join(important_ents)
        return f'''\n{LongDocPrompt._context_w_note_format(recap, passage)}\n\nImportant entities:\n{important_ents_str}\n\nAbove is a recap of several previous passages, the current passage and the important entities in the current passage.\nMake use of the recap information to help you better understand the current passage.\nFind the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the current passage.\nTry to include as many entities as possible in each cluster.\n\n{LongDocPrompt._relation_description_format()}\n'''

    @staticmethod
    def shorten_w_note(recap:str, passage:str):
        """
        Given the recap of several previous passages and the current passage, please shorten the current passage.
        Make use of the recap information to help you better understand the current passage.
        Just give me a shortened version. DO NOT explain your reason.

        {_context_w_note_format}
        """
        return f"""\nGiven the recap of several previous passages and the current passage, please shorten the current passage.\nMake use of the recap information to help you better understand the current passage.\nJust give me a shortened version. DO NOT explain your reason.\n\n{LongDocPrompt._context_w_note_format(recap, passage)}\n"""

    @staticmethod
    def embed_w_note(recap:str, doc_type:str):
        """
        Use the recap of several previous passages to help you understand the current {doc_type}.

        Recap:
        {recap}
        
        Current {doc_type.capitalize()}:
        """
        return f"""\nUse the recap of several previous passages to help you understand the current {doc_type}.\n\nRecap:\n{recap}\n\nCurrent {doc_type.capitalize()}:\n"""