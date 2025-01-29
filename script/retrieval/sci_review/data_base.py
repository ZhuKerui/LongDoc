from .base import *

from pydantic import BaseModel
import requests
import evaluate
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize, ngrams

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

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"File '{filename}' downloaded successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        
        
# Dataset Sample
class Sample(BaseModel):
    doc_file: str = ''
    doc_str: str = ''
    doc_strs: list[str] = []
    outline: str = ''
    question_types: list[str] = []
    questions: dict[str, str] = {}
    answers: dict[str, str] = {}
    extractions: dict[str, list[str]] = {}
    
    
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
        sent_ngrams = {ngram for n in range(2, 6) for ngram in ngrams(tokenized_sent, n)}
        sent_unique_ngrams = sent_ngrams.intersection(unique_ngram2sent)
        if sent_unique_ngrams:
            sent_ids.append(unique_ngram2sent[sent_unique_ngrams.pop()][0])
    return sent_ids

def get_binary_sent_ids(sents:list[str], unique_ngram2sent:dict[tuple, tuple[int, str]]):
    sent_ids = [0] * (max(sent[0] for ngram, sent in unique_ngram2sent.items()) + 1)
    for sent in sents:
        tokenized_sent = word_tokenize(sent)
        sent_ngrams = {ngram for n in range(2, 6) for ngram in ngrams(tokenized_sent, n)}
        sent_unique_ngrams = sent_ngrams.intersection(unique_ngram2sent)
        if sent_unique_ngrams:
            sent_ids[unique_ngram2sent[sent_unique_ngrams.pop()][0]] = 1
    return sent_ids