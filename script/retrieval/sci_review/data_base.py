from .base import *

from pydantic import BaseModel
from bs4 import Tag, BeautifulSoup
import requests
import urllib.request as libreq
import feedparser
import evaluate
import re
from time import sleep
from nltk.tokenize import word_tokenize
from nltk import ngrams
from spacy import Language

ACL = 'ACL'
ARXIV = 'ArXiv'

class EvalMetrics:
    def __init__(self):
        self.rouge = evaluate.load('rouge')
        self.bertscore = evaluate.load("bertscore")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load('recall')
        self.f1 = evaluate.load('f1')
    
    def eval_rouge(self, predictions:list[str], references:list[list[str]]):
        return self.rouge.compute(predictions=predictions, references=references)
    
    def eval_bertscore(self, predictions:list[str], references:list[list[str]]):
        return self.bertscore.compute(predictions=predictions, references=references, lang="en")
    
    def eval_precision_recall_f1(self, predictions:list[int], references:list[int]):
        return {
            **self.precision.compute(predictions=predictions, references=references),
            **self.recall.compute(predictions=predictions, references=references),
            **self.f1.compute(predictions=predictions, references=references)
        }


def download_file(url, filename):
    """Downloads a file from a given URL and saves it with the specified filename."""

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception if the request failed

        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"File '{filename}' downloaded successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        
def get_arxiv_id_by_title(title:str, max_results:int=1):
    non_ascii_chars = {char for char in title if not char.isascii()}
    if '’' in non_ascii_chars:
        non_ascii_chars.remove('’')
        title = title.replace('’', "'")
    for char in non_ascii_chars:
        title = title.replace(char, ' ')
    
    title = ' '.join(title.split())
    
    search_query = f'ti:{title}'.replace(' ', '+') # search for electron in all fields
    query = f'search_query={search_query}&start=0&max_results={max_results}'

    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'
    with libreq.urlopen(base_url+query) as url:
        response:bytes = url.read()

    # parse the response using feedparser
    feed = feedparser.parse(response)

    # Run through each entry, and print out information
    for entry in feed.entries:
        arxiv_id:str = entry.id.split('/abs/')[-1]
        arxiv_title:str = entry.title
        if ''.join(arxiv_title.split()).lower() == ''.join(title.split()).lower():
            return arxiv_id
        
def get_paper_ids_by_semantic_scholar(corpus_id:int=None, paper_id:str=None, acl_id:str=None):
    assert corpus_id is not None or paper_id is not None or acl_id is not None, "At least one of corpus_id, paper_id, or acl_id must be provided."
    base_url = "https://api.semanticscholar.org/graph/v1/paper/"
    if corpus_id is not None:
        link = base_url + f"CorpusID:{corpus_id}"
    elif paper_id is not None:
        link = base_url + f"{paper_id}"
    elif acl_id is not None:
        link = base_url + f"ACL:{acl_id}"
        
    paper_meta = requests.get(link, params={'fields': 'externalIds'}).json()
    while 'externalIds' not in paper_meta:
        sleep(10)
        paper_meta = requests.get(link, params={'fields': 'externalIds'}).json()
    
    return {
        ARXIV: paper_meta['externalIds'].get(ARXIV, None), 
        ACL: paper_meta['externalIds'].get(ACL, None)
    }
    
def download_arxiv_pdf(arxiv_id:str, filename:str):
    link = f"https://arxiv.org/pdf/{arxiv_id}"
    download_file(link, filename)

def download_acl_pdf(acl_id:str, filename:str):
    link = f'https://aclanthology.org/{acl_id}.pdf'
    download_file(link, filename)
    
def get_arxiv_paper_text(arxiv_id:str):
    link = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    response = requests.get(link, verify=False)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    paragraphs = list[str]()
    headers = []
    title_text = ''
    p:Tag
    for p in soup.find_all(class_={'ltx_para', 'ltx_title', 'ltx_abstract'}):
        if 'ltx_title_bibliography' in p['class'] or 'ltx_title_acknowledgements' in p['class']:
            break
        is_paragraph = p.name == 'div' and 'ltx_para' in p['class'] and re.match(r"^S\d+\.+(p|S)", p['id'])
        is_header = re.fullmatch(r"^h\d+", p.name) is not None
        is_abstract = 'ltx_abstract' in p['class']
        is_abstract_title = 'ltx_title_abstract' in p['class']
        if is_paragraph or is_header or is_abstract or is_abstract_title:
            block_text:str = p.text
            for note in p.find_all(class_= 'ltx_note'):
                block_text = block_text.replace(note.text, '')
            for math_note in p.find_all(class_= 'ltx_Math'):
                annotation = math_note.find('annotation')
                if annotation is not None:
                    block_text = block_text.replace(math_note.text, annotation.text)
            if not is_abstract:
                block_text = re.sub(r"\n", "", block_text)
                block_text = ' '.join(block_text.split())
            
            if p.name == 'h1':
                title_text = block_text
            elif is_abstract_title:
                abstract_title = (block_text, p.name)
            elif is_abstract:
                abstract_text = ''.join(block_text.split("\n")[2:])
            else:
                paragraphs.append(block_text)
                if is_header:
                    headers.append((block_text, p.name))
    paragraphs.insert(0, abstract_text)
    paragraphs.insert(0, abstract_title[0])
    headers.insert(0, abstract_title)
    header_size_min = min(int(header[1][1:]) for header in headers)
    outline = '\n'.join(('    ' * ((int(header_size[1:]) - header_size_min) if hid > 0 else 0)) + header for hid, (header, header_size) in enumerate(headers))
    return paragraphs, outline, title_text
    
# Dataset Sample
class Sample(BaseModel):
    doc_file: str = '' # ACL:$ or ArXiv:$
    doc_str: str = ''
    doc_strs: list[str] = []
    outline: str = ''
    question_types: list[str] = []
    questions: dict[str, list[str]] = {}
    question_meta: dict[str, dict] = {}
    answers: dict[str, str] = {}
    extractions: dict[str, list[str]] = {}
    
class Extraction(BaseModel):
    chunks: list[str] = []
    extractions: list[str] = []
    chunk_sources: list[list[int]] = []
    
def get_sent_index(sents:list[str]):
    # sents = [sent for p in text.split(PARAGRAPH_SEP) for sent in sent_tokenize(p)]
    ngram2sents: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for sid, sent in enumerate(sents):
        tokenized_sent = word_tokenize(sent)
        for n in range(2, 8):
            for ngram in ngrams(tokenized_sent, n):
                ngram2sents[ngram].append((sid, sent))

    unique_ngram2sent = {ngram:sents[0] for ngram, sents in ngram2sents.items() if len(sents) == 1}
    return unique_ngram2sent


def get_sent_ids(sents:list[str], unique_ngram2sent):
    sent_ids = list[int]()
    for sent in sents:
        tokenized_sent = word_tokenize(sent)
        sent_ngrams = {ngram for n in range(2, 8) for ngram in ngrams(tokenized_sent, n)}
        sent_unique_ngrams = sent_ngrams.intersection(unique_ngram2sent)
        sent_ids.append(unique_ngram2sent[sent_unique_ngrams.pop()][0] if sent_unique_ngrams else -1)
    return sent_ids

def get_binary_sent_ids(sents:list[str], unique_ngram2sent:dict[tuple, tuple[int, str]]):
    sent_ids = [0] * (max(sent[0] for ngram, sent in unique_ngram2sent.items()) + 1)
    for sent in sents:
        tokenized_sent = word_tokenize(sent)
        sent_ngrams = {ngram for n in range(2, 8) for ngram in ngrams(tokenized_sent, n)}
        sent_unique_ngrams = sent_ngrams.intersection(unique_ngram2sent)
        if sent_unique_ngrams:
            sent_ids[unique_ngram2sent[sent_unique_ngrams.pop()][0]] = 1
    return sent_ids

def spacy_sent_tokenize(nlp:Language, text:str):
    return [sent.text for sent in nlp(text, disable=["lemmatizer", "ner", "positionrank", 'fastcoref']).sents]