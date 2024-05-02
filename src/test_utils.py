import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from typing import List, Tuple
from nltk import sent_tokenize, word_tokenize

def split_sents(retriever_tokenizer:AutoTokenizer, p_input_ids:np.ndarray, is_contriever:bool):
    sents = sent_tokenize(retriever_tokenizer.decode(p_input_ids[1:-1] if is_contriever else p_input_ids))
    sent_lens = [len(retriever_tokenizer.encode(sent)) - 2 for sent in sents]
    sent_start = 1 if is_contriever else 0
    sent_spans = []
    for sid in range(len(sents)):
        sent_end = sent_start + sent_lens[sid]
        while len(retriever_tokenizer.decode(p_input_ids[sent_start:sent_end]).strip()) < len(sents[sid]):
            sent_end += 1
        sent_spans.append((sent_start, sent_end))
        sent_start = sent_end
    return sent_spans

def important_page_tokens(retriever_tokenizer:AutoTokenizer, question:str, pages, q_lhs:np.ndarray, q_input_ids:np.ndarray, q_emb, p_lhs, p_input_ids, pids, scores):
    print(question)
    for i in range(q_lhs.shape[0]):
        print(retriever_tokenizer.decode(q_input_ids[i]), np.linalg.norm(q_lhs[i]))
    print('\n')
    for rank, (pid, score) in enumerate(zip(pids, scores)):
        print(f'Rank {rank}\nPassage {pid}:\n{score}\n')
        print(pages[pid])
        token_scores = p_lhs[pid].dot(q_emb)
        max_indices = np.argsort(token_scores)[::-1][:(token_scores>2).sum()].tolist()
        print('\n\nHigh scored spans:\n')
        for idx in max_indices:
            print(token_scores[idx], f'<{retriever_tokenizer.decode(p_input_ids[pid][max(0, idx - 1): idx + 1])}>', retriever_tokenizer.decode(p_input_ids[pid][max(0, idx - 5): idx + 5]))
        print('\n\n')

def query_indicatiors(retriever_tokenizer:AutoTokenizer, question:str, pages, q_lhs:np.ndarray, q_input_ids:np.ndarray, p_lhs, p_input_ids, pids, scores, top_k:int=None, q_spans:List[Tuple[int, int]]=None):
    print(question)
    x = []
    if q_spans:
        new_q_lhs = []
        for sid, span in enumerate(q_spans):
            span_lhs = q_lhs[span[0] : span[1]].mean(axis=0)
            new_q_lhs.append(span_lhs)
            token = retriever_tokenizer.decode(q_input_ids[span[0] : span[1]])
            print(token, np.linalg.norm(span_lhs))
            x.append(f'{token}_{sid}')
        q_lhs = np.vstack(new_q_lhs)
    else:
        for tid, t in enumerate(q_input_ids):
            token = retriever_tokenizer.decode(t)
            print(token, np.linalg.norm(q_lhs[tid]))
            x.append(f'{token}_{tid}')
        
    print('\n')
    for pid, score in zip(pids, scores):
        print(f'{pid}: {score}')
    for rank, (pid, score) in enumerate(zip(pids, scores)):
        q_token_scores = np.matmul(q_lhs, p_lhs[pid].T)
        y = [np.sort(token_scores)[::-1][:top_k].mean() for token_scores in q_token_scores]
        y.reverse()
        plt.barh(list(reversed(x)), y)
        for tid, t_score in enumerate(y):
            plt.text(t_score, tid, str(t_score)[:4])
        plt.show()
        print(f'Rank {rank}\nPassage {pid}:\n{score}\n')
        print(pages[pid])
        print('\n\nHigh scored spans:\n')
        for q_token, token_scores in zip(x, q_token_scores):
            max_indices = np.argsort(token_scores)[::-1].tolist()[:10]
            print(np.sort(token_scores)[::-1][:top_k].mean(), 
                  f'<{q_token}>', 
                  *[(token_scores[idx], f'<{retriever_tokenizer.decode(p_input_ids[pid][max(0, idx - 5): idx + 1])}>') for idx in max_indices],
                  '\n')
        
def query_indicator_sents(retriever_tokenizer:AutoTokenizer, pages, q_lhs:np.ndarray, q_input_ids:np.ndarray, p_lhs, p_input_ids, test_pid, test_q_token_id:int, is_contriever:bool):
    print(retriever_tokenizer.decode(q_input_ids[test_q_token_id]), '\n')
    print(pages[test_pid], '\n')
    sent_spans = split_sents(retriever_tokenizer, p_input_ids[test_pid], is_contriever)
    scores = p_lhs[test_pid].dot(q_lhs[test_q_token_id])
    sent_scores = [(scores[sent_span[0]:sent_span[1]].mean(), retriever_tokenizer.decode(p_input_ids[test_pid][sent_span[0]:sent_span[1]])) for sent_span in sent_spans]
    sent_scores.sort(key=lambda x: x[0], reverse=True)
    for score, sent in sent_scores:
        print(score, sent)
        
def decode_span(input_ids:List[int], tokenizer:AutoTokenizer, spans:List[Tuple[int, int]]):
    return [tokenizer.decode(input_ids[w_span[0] : w_span[1]]) for w_span in spans]

def word_split(input_ids:List[int], tokenizer:AutoTokenizer, skip_start:bool=False, skip_end:bool=False):
    if skip_start:
        input_ids = input_ids[1:]
    if skip_end:
        input_ids = input_ids[:-1]
    words:List[str] = word_tokenize(tokenizer.decode(input_ids))
    wid = 0
    w_spans = []
    temp_w_span = []
    w_start = 0
    for tid, input_id in enumerate(input_ids):
        temp_w_span.append(input_id)
        temp_str = tokenizer.decode(temp_w_span)
        if sum([len(t) for t in temp_str.split()]) == len(words[wid]):
            w_spans.append([w_start, tid + 1])
            temp_w_span = []
            w_start = tid + 1
            wid += 1
    if skip_start:
        w_spans = [(span[0] + 1, span[1] + 1) for span in w_spans]
    return w_spans