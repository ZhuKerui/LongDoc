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
To read a page again, respond with “Page $PAGE_NUM”, replacing $PAGE_NUM with the target page number.
You can only specify a SINGLE page in your response at this time.
Pages {} are re-read already. DO NOT ask to read them again.
To stop, simply say “STOP”.
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