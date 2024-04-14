prompt_pagination_template = """
You are given a passage that is taken from a larger text (article, book, ...) and some numbered labels between the paragraphs in the passage.
Numbered label are in angeled brackets. For example, if the label number is 19, it shows as <19> in text.
Please choose one label that it is natural to break reading.
Such point can be scene transition, end of a dialogue, end of an argument, narrative transition, etc.
Please answer the break point label and explain.
For example, if <57> is a good point to break, answer with \"Break point: <57>\n Because ...\"

Passage:

{0}
{1}
{2}

"""

prompt_shorten_template = """
Please shorten the following passage.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{}

"""

prompt_parallel_lookup_template = """
The following text is what you remembered from reading an article and a question related to it.
You may read 5 page(s) of the article again to refresh your memory to prepare yourselve for the question.
Please respond with which page(s) you would like to read.
For example, if your only need to read Page 8, respond with \"I want to look up Page [8] to ...\";
if your would like to read Page 7 and 12, respond with \"I want to look up Page [7, 12] to ...\";
if your would like to read Page 2, 3, 7, 15 and 18, respond with \"I want to look up Page [2, 3, 7, 15, 18] to ...\".
if your would like to read Page 3, 4, 5, 12, 13 and 16, respond with \"I want to look up Page [3, 3, 4, 12, 13, 16] to ...\".
DO NOT answer the question yet.

Text:
{}

Question:
{}

Take a deep breath and tell me: Which 5 page(s) would you like to read again?
"""

prompt_sequential_lookup_template = """
The following text is what you remember from reading a meeting transcript, followed by a question about the transcript.
You may read multiple pages of the transcript again to refresh your memory and prepare to answer the question.

Text:
{}

Question:
{}

Please specify a SINGLE page you would like to read again to gain more information or say "STOP" if the information is adequate to answer the question.
For example, if your would like to read Page 7, respond with \"I want to look up Page 7 to ...\";
if your would like to read Page 13, respond with \"I want to look up Page 13 to ...\";
if you think the information is adequate, simply respond with \"STOP\".
Pages {} have been looked up already and DO NOT ask to read them again.
DO NOT answer the question in your response.
"""

prompt_answer_template = '''
Read the following article and answer a {question_type}.

Article:
{article}

Question:
{question}

{answer_format}
'''

prompt_ent_description_template = '''
Passage:
{paragraph}

Above is part of a {context_type}. Based on the passage, briefly and truthfully describe the information of each following entity:
{important_ents_str}

Generate your response using the following format:
"{important_ents_0}: the information of {important_ents_0}\n{important_ents_1}: the information of {important_ents_1}\n..."
'''

prompt_relation_description_template = '''
Passage:
{paragraph}

Important entities:
{important_ents_str}

Above is part of a {context_type} and the important entities in the passage.
Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the above passage.
Try to include as many entities as possible in each cluster.
Generate your response in the following format:
"Relation summary:\n1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.\n2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.\n..."
'''

prompt_shorten_w_note_template = """
Given the recap of several previous passages and the current passage, please shorten the current passage.
Make use of the recap information to help you better understand the current passage.
Just give me a shortened version. DO NOT explain your reason.

Recap:
{}

Current Passage:
{}

"""

prompt_ent_description_w_note_template = '''
Recap:
{recap}

Current Passage:
{paragraph}

Above is a recap of several previous passages and the content of the current passage of a {context_type}.
Make use of the recap information to help you better understand the current passage.
Based on the recap and the current passage, briefly and truthfully describe the information of each following entity in the current passage:

{important_ents_str}

For each entity information,
if it includes any information from the recap, cite the source passages like "Entity 1: the information of Entity 1. (Recap: Passage -1, Passage -2, ...)";
if it is totally based on the current passage, cite the source passage like "Entity 2: the information of Entity 2. (Current passage)".
'''

prompt_relation_description_w_note_template = '''
Recap:
{recap}

Current Passage:
{paragraph}

Important entities:
{important_ents_str}

Above is a recap of several previous passages, the content of the current passage of a {context_type} and the important entities in the current passage.
Make use of the recap information to help you better understand the current passage.
Find the related important entity clusters and use 1 to 3 sentences to informatively summarize their relational information in the current passage.
Try to include as many entities as possible in each cluster.
Generate your response in the following format:
"Relation summary:\n1. (Entity 1, Entity 2, Entity 3): summary of relational information between Entity 1, 2 and 3.\n2. (Entity 2, Entity 4): summary of relational information between Entity 2 and 4.\n..."
'''