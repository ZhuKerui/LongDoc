import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from typing import List, Tuple
from nltk import sent_tokenize, word_tokenize
import seaborn as sb

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

def merge_words_and_embeddings(retriever_tokenizer:AutoTokenizer, input_ids:np.ndarray, lhs:np.ndarray, word_spans:List[Tuple[int, int]], verbose:bool=True):
    strs:List[str] = []
    if word_spans:
        new_lhs = []
        for sid, span in enumerate(word_spans):
            span_lhs = lhs[span[0] : span[1]].mean(axis=0)
            new_lhs.append(span_lhs)
            token = retriever_tokenizer.decode(input_ids[span[0] : span[1]])
            if verbose:
                print(token, np.linalg.norm(span_lhs))
            strs.append(f'{token}_{sid}')
        lhs = np.vstack(new_lhs)
    else:
        for tid, t in enumerate(input_ids):
            token = retriever_tokenizer.decode(t)
            if verbose:
                print(token, np.linalg.norm(lhs[tid]))
            strs.append(f'{token}_{tid}')
    return strs, lhs
            
def query_indicatiors(question:str, pages, q_strs:List[str], q_lhs:np.ndarray, p_lhs, p_strs:List[List[str]], pids, scores, top_k:int=None):
    print(question)
    print('\n')
    for pid, score in zip(pids, scores):
        print(f'{pid}: {score}')
    for rank, (pid, score) in enumerate(zip(pids, scores)):
        q_token_scores = np.matmul(q_lhs, p_lhs[pid].T)
        top_scores = [np.sort(token_scores)[::-1][:top_k] for token_scores in q_token_scores]
        y = [top_score.sum() / (top_score > 0).sum() for top_score in top_scores]
        y.reverse()
        plt.barh(list(reversed(q_strs)), y)
        for tid, t_score in enumerate(y):
            plt.text(t_score, tid, str(t_score)[:4])
        plt.show()
        print(f'Rank {rank}\nPassage {pid}:\n{score}\n')
        print(pages[pid])
        print('\n\nHigh scored spans:\n')
        for tid, t_score in enumerate(y):
            max_indices = np.argsort(q_token_scores[tid])[::-1].tolist()[:10]
            print(t_score, 
                  f'<{q_strs[tid]}>', 
                  *[(q_token_scores[tid, idx], f'<{" ".join(p_strs[pid][max(0, idx - 5): idx + 1])}>') for idx in max_indices],
                  '\n')

def query_distribution(q_strs:List[str], q_lhs:np.ndarray, p_lhs, top_k:int=None):
    ys = []
    for lhs in p_lhs:
        q_token_scores = np.matmul(q_lhs, lhs.T)
        top_scores = [np.sort(token_scores)[::-1][:top_k] for token_scores in q_token_scores]
        y = [top_score.sum() / (top_score > 0).sum() for top_score in top_scores]
        ys.append(y)
    sb.heatmap(np.vstack(ys).T, yticklabels=q_strs)
    plt.show()
        
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

def word_split(input_ids:List[int], tokenizer:AutoTokenizer, bos:str='', eos:str=''):
    input_str = tokenizer.decode(input_ids)
    if bos:
        input_str = input_str[len(bos):]
        input_ids = input_ids[1:]
    if eos:
        input_str = input_str[:-len(eos)]
        input_ids = input_ids[:-1]
    words:List[str] = word_tokenize(input_str)
    wid = 0
    w_spans = []
    temp_w_span = []
    t_start = 0
    w_end = 1
    for tid, input_id in enumerate(input_ids):
        temp_w_span.append(input_id)
        temp_str = tokenizer.decode(temp_w_span)
        if sum([len(t) for t in temp_str.split()]) == sum([len(w) for w in words[wid:w_end]]):
            w_spans.append([t_start, tid + 1])
            temp_w_span = []
            t_start = tid + 1
            wid = w_end
            w_end += 1
        elif sum([len(t) for t in temp_str.split()]) > sum([len(w) for w in words[wid:w_end]]):
            w_end += 1
    if bos:
        w_spans = [(0, 1)] + [(span[0] + 1, span[1] + 1) for span in w_spans]
    if eos:
        last_span = w_spans[-1]
        w_spans.append((last_span[1], last_span[1] + 1))
    return w_spans