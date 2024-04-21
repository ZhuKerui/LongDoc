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
    def list_entity(passage):
        '''
        Passage:
        {passage}

        List the important named entities in the above passage that are relevant to most of its content.
        Don't give any explanation.
        Generate your response in the following format:
        "Important entities:
        1. Entity 1
        2. Entity 2
        3. Entity 3
        ..."
        '''
        return f'''\nPassage:\n{passage}\n\nList the important named entities in the above passage that are relevant to most of its content.\nDon't give any explanation.\nGenerate your response in the following format:\n"Important entities:\n1. Entity 1\n2. Entity 2\n3. Entity 3\n..."\n'''
    
    @staticmethod
    def list_entity_w_note(recap, passage):
        '''
        Recap:
        {recap}
        
        Current Passage:
        {passage}

        Above is a recap of several previous passages and the content of the current passage.
        Make use of the recap information to help you better understand the current passage.
        Based on the recap and the current passage, list the important named entities in the current passage that are relevant to most of its content.
        Don't give any explanation.
        Generate your response in the following format:
        "Important entities:
        1. Entity 1
        2. Entity 2
        3. Entity 3
        ..."
        '''
        return f'''\nRecap:\n{recap}\n\nCurrent Passage:\n{passage}\n\nAbove is a recap of several previous passages and the content of the current passage.\nMake use of the recap information to help you better understand the current passage.\nBased on the recap and the current passage, list the important named entities in the current passage that are relevant to most of its content.\nDon't give any explanation.\nGenerate your response in the following format:\n"Important entities:\n1. Entity 1\n2. Entity 2\n3. Entity 3\n..."\n'''
        
    @staticmethod
    def ent_description(passage, important_ents_str, important_ents_0, important_ents_1):
        '''
        Context:
        {passage}

        Based on the above context, briefly and truthfully describe the information of each following entity:
        {important_ents_str}

        Generate your response using the following format:
        "{important_ents_0}: the information of {important_ents_0}
        {important_ents_1}: the information of {important_ents_1}
        ..."
        '''
        return f'''\nContext:\n{passage}\n\nBased on the above context, briefly and truthfully describe the information of each following entity:\n{important_ents_str}\n\nGenerate your response using the following format:\n"{important_ents_0}: the information of {important_ents_0}\n{important_ents_1}: the information of {important_ents_1}\n..."\n'''
    
    @staticmethod
    def relation_description(passage, important_ents_str):
        '''
        Context:
        {passage}

        Important entities:
        {important_ents_str}

        Above is a context and the important entities in the context.
        Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above passage.
        Try to include as many entities as possible in each cluster.
        Generate your response in the following format:
        "Relation summary:
        1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.
        2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.
        ..."
        '''
        return f'''\nContext:\n{passage}\n\nImportant entities:\n{important_ents_str}\n\nAbove is a context and the important entities in the context.\nFind the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above passage.\nTry to include as many entities as possible in each cluster.\nGenerate your response in the following format:\n"Relation summary:\n1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.\n2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.\n..."\n'''

    @staticmethod
    def shorten_w_note(recap:str, passage:str):
        """
        Given the recap of several previous passages and the current passage, please shorten the current passage.
        Make use of the recap information to help you better understand the current passage.
        Just give me a shortened version. DO NOT explain your reason.

        Recap:
        {recap}

        Current Passage:
        {passage}

        """
        return f"""\nGiven the recap of several previous passages and the current passage, please shorten the current passage.\nMake use of the recap information to help you better understand the current passage.\nJust give me a shortened version. DO NOT explain your reason.\n\nRecap:\n{recap}\n\nCurrent Passage:\n{passage}\n\n"""

    @staticmethod
    def ent_description_w_note(recap, passage, important_ents_str):
        '''
        Recap:
        {recap}

        Current Passage:
        {passage}

        Above is a recap of several previous passages and the content of the current passage.
        Make use of the recap information to help you better understand the current passage.
        Based on the recap and the current passage, briefly and truthfully describe the information of each following entity in the current passage:

        {important_ents_str}

        For each entity information,
        if it includes any information from the recap, cite the source passages like "Entity 1: the information of Entity 1. (Recap: Passage -1, Passage -2, ...)";
        if it is totally based on the current passage, cite the source passage like "Entity 2: the information of Entity 2. (Current passage)".
        '''
        return f'''\nRecap:\n{recap}\n\nCurrent Passage:\n{passage}\n\nAbove is a recap of several previous passages and the content of the current passage.\nMake use of the recap information to help you better understand the current passage.\nBased on the recap and the current passage, briefly and truthfully describe the information of each following entity in the current passage:\n\n{important_ents_str}\n\nFor each entity information,\nif it includes any information from the recap, cite the source passages like "Entity 1: the information of Entity 1. (Recap: Passage -1, Passage -2, ...)";\nif it is totally based on the current passage, cite the source passage like "Entity 2: the information of Entity 2. (Current passage)".\n'''

    @staticmethod
    def relation_description_w_note(recap, passage, important_ents_str):
        '''
        Recap:
        {recap}

        Current Passage:
        {passage}

        Important entities:
        {important_ents_str}

        Above is a recap of several previous passages, the content of the current passage and the important entities in the current passage.
        Make use of the recap information to help you better understand the current passage.
        Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the current passage.
        Try to include as many entities as possible in each cluster.
        Generate your response in the following format:
        "Relation summary:
        1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.
        2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.
        ..."
        '''
        return f'''\nRecap:\n{recap}\n\nCurrent Passage:\n{passage}\n\nImportant entities:\n{important_ents_str}\n\nAbove is a recap of several previous passages, the content of the current passage and the important entities in the current passage.\nMake use of the recap information to help you better understand the current passage.\nFind the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the current passage.\nTry to include as many entities as possible in each cluster.\nGenerate your response in the following format:\n"Relation summary:\n1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.\n2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.\n..."\n'''