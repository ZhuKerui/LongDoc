import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from typing import List, Tuple

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
    x.reverse()
        
    print('\n')
    for pid, score in zip(pids, scores):
        print(f'{pid}: {score}')
    for rank, (pid, score) in enumerate(zip(pids, scores)):
        q_token_scores = np.matmul(q_lhs, p_lhs[pid].T)
        y = [np.sort(token_scores)[::-1][:top_k].mean() for token_scores in q_token_scores]
        y.reverse()
        plt.barh(x, y)
        plt.show()
        print(f'Rank {rank}\nPassage {pid}:\n{score}\n')
        print(pages[pid])
        print('\n\nHigh scored spans:\n')
        for q_token, token_scores in zip(x, q_token_scores):
            max_indices = np.argsort(token_scores)[::-1].tolist()[:10]
            print(np.sort(token_scores)[::-1][:top_k].mean(), f'<{q_token}>', *[(token_scores[idx], f'<{retriever_tokenizer.decode(p_input_ids[pid][max(0, idx - 5): idx + 1])}>') for idx in max_indices])
        print('\n\n')