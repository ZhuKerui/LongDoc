import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from typing import List, Tuple
from nltk import sent_tokenize, word_tokenize
import seaborn as sb
from sklearn.manifold import TSNE
import pandas as pd
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

from .base_utils import ChunkInfo
from .models import Retriever

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

def merge_words_and_embeddings(retriever_tokenizer:AutoTokenizer, input_ids:np.ndarray, lhs:np.ndarray, word_spans:List[Tuple[int, int]], verbose:bool=True, indexed:bool=True):
    strs:List[str] = []
    if word_spans:
        new_lhs = []
        for sid, span in enumerate(word_spans):
            span_lhs = lhs[span[0] : span[1]].mean(axis=0)
            new_lhs.append(span_lhs)
            token = retriever_tokenizer.decode(input_ids[span[0] : span[1]])
            if verbose:
                print(token, np.linalg.norm(span_lhs))
            strs.append(f'{token}_{sid}' if indexed else token)
        lhs = np.vstack(new_lhs)
    else:
        for tid, t in enumerate(input_ids):
            token = retriever_tokenizer.decode(t)
            if verbose:
                print(token, np.linalg.norm(lhs[tid]))
            strs.append(f'{token}_{tid}' if indexed else token)
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
        
def decode_span(input_ids:List[int], tokenizer:AutoTokenizer, spans:List[Tuple[int, int]]):
    return [tokenizer.decode(input_ids[w_span[0] : w_span[1]]) for w_span in spans]

def word_split(input_ids:List[int], tokenizer:AutoTokenizer, bos:str='', eos:str=''):
    input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]
    input_str = tokenizer.decode(input_ids)
    if bos:
        input_str = input_str[len(bos):]
        input_ids = input_ids[1:]
    if eos:
        input_str = input_str[:-len(eos)]
        input_ids = input_ids[:-1]
    words = [w if w not in ['``', "''"] else '"' for w in word_tokenize(input_str)]
    wid = 0
    w_spans = []
    temp_w_span = []
    t_start = 0
    w_end = 1
    for tid, input_id in enumerate(input_ids):
        temp_w_span.append(input_id)
        temp_str_nw, temp_word_str_nw = ''.join(tokenizer.decode(temp_w_span).split()), ''.join(words[wid:w_end])
        if temp_str_nw == temp_word_str_nw:
            w_spans.append([t_start, tid + 1])
            temp_w_span = []
            t_start = tid + 1
            wid = w_end
            w_end += 1
        elif len(temp_str_nw) > len(temp_word_str_nw):
            w_end += 1
    if wid != len(words):
        w_spans.append([t_start, len(input_ids)])
    if bos:
        w_spans = [(0, 1)] + [(span[0] + 1, span[1] + 1) for span in w_spans]
    if eos:
        last_span = w_spans[-1]
        w_spans.append((last_span[1], last_span[1] + 1))
    return w_spans

def sent_split(input_ids:List[int], tokenizer:AutoTokenizer, bos:str='', eos:str=''):
    input_ids = [i for i in input_ids if i != tokenizer.pad_token_id]
    input_str = tokenizer.decode(input_ids)
    if bos:
        input_str = input_str[len(bos):]
        input_ids = input_ids[1:]
    if eos:
        input_str = input_str[:-len(eos)]
        input_ids = input_ids[:-1]
    sents:List[str] = sent_tokenize(input_str)
    s_spans = []
    temp_s_span = []
    t_start = 0
    sid = 0
    words = []
    for tid, input_id in enumerate(input_ids):
        if not words:
            words = [w if w not in ['``', "''"] else '"' for w in word_tokenize(sents[sid])]
            sent_str = ''.join(words)
            sent_start_str = sent_str[:20]
            s_started = False
            t_offset = -1
            s_offset = -1
        temp_s_span.append(input_id)
        temp_str_nw = ''.join(tokenizer.decode(temp_s_span).split())
        if not s_started:
            overlap_num = len(set(temp_str_nw).intersection(set(sent_start_str)))
            if overlap_num > 0:
                for i in range(len(temp_str_nw)):
                    if temp_str_nw[i:] in sent_start_str:
                        t_offset = i
                        s_offset = sent_start_str.index(temp_str_nw[t_offset:])
                        break
            s_started = t_offset >= 0
            if not s_started:
                temp_s_span = []
            else:
                t_start = tid
        else:
            if sent_str[s_offset:] in temp_str_nw:
                s_spans.append([t_start, tid + 1])
                words = []
                temp_s_span = []
                sid += 1
    if bos:
        s_spans = [(span[0] + 1, span[1] + 1) for span in s_spans]
    assert len(s_spans) == len(sents)
    return s_spans

def tsne_plot(embeddings:np.ndarray, perplexity=4):
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=perplexity, metric='cosine').fit_transform(embeddings)
    df = pd.DataFrame(X_embedded, columns=['x', 'y'])
    ax = sb.scatterplot(df, x='x', y='y')
    for pid, (x, y) in enumerate(X_embedded):
        ax.text(x+0.01, y, str(pid))

def print_input_ids(p_strs:List[List[str]], iids:List[int]):
    for iid in iids:
        print(iid, ' '.join(p_strs[iid]))
        
def print_pages(pages:List[str], iids:List[int]):
    for iid in iids:
        print(iid, pages[iid])
        
def plot_score_matrix(retriever_tokenizer:AutoTokenizer, x_input_ids:np.ndarray, x_lhs:np.ndarray, x_word_spans:List[Tuple[int, int]], y_input_ids:np.ndarray, y_lhs:np.ndarray, y_word_spans:List[Tuple[int, int]], indexed:bool=True, worded:bool=True):
    x_norm, y_norm = np.linalg.norm(x_lhs.mean(0)), np.linalg.norm(y_lhs.mean(0))
    x_lhs = x_lhs / x_norm
    y_lhs = y_lhs / y_norm
    token_score_mat = y_lhs @ x_lhs.T
    if not x_word_spans:
        x_word_spans = list(zip(range(len(x_input_ids)), range(1, len(x_input_ids)+1)))
    if not y_word_spans:
        y_word_spans = list(zip(range(len(y_input_ids)), range(1, len(y_input_ids)+1)))
    score_mat = np.zeros((len(y_word_spans), len(x_word_spans)))
    x_strs = [retriever_tokenizer.decode(x_input_ids[span[0] : span[1]]) + (f'_{sid}' if indexed else '') for sid, span in enumerate(x_word_spans)]
    y_strs = [retriever_tokenizer.decode(y_input_ids[span[0] : span[1]]) + (f'_{sid}' if indexed else '') for sid, span in enumerate(y_word_spans)]
    print('x:', x_strs)
    print('y:', y_strs)
    for xid, x_span in enumerate(x_word_spans):
        for yid, y_span in enumerate(y_word_spans):
            score_mat[yid, xid] = token_score_mat[y_span[0]:y_span[1], x_span[0]:x_span[1]].mean()
    fig, ax = plt.subplots(figsize=(score_mat.shape[1], score_mat.shape[0]))
    sb.heatmap(score_mat, xticklabels=range(score_mat.shape[1]) if not worded else x_strs, yticklabels=range(score_mat.shape[0]) if not worded else y_strs, annot=True, ax=ax)

def plot_map(results:List[ChunkInfo], queries:List[str], retriever:Retriever):
    raw_symbols = SymbolValidator().values
    markers = [raw_symbols[i+2] for i in range(0,len(raw_symbols),3) if '-' not in raw_symbols[i+2]]
    width = len(results)
    height = max([len(ci.important_ents) for ci in results])
    node_x = []
    node_y = []
    nodes = []
    node_pos = []
    for cid, ci in enumerate(results):
        temp_node_pos = {}
        important_ents = ci.important_ents
        important_ents.sort()
        for eid, ent in enumerate(important_ents):
            x, y = cid, (height / 2) + np.sign(eid - len(important_ents) / 2) * (np.log(np.abs(eid - len(important_ents) / 2) + 1) + np.log(cid % 3 + 1))
            node_x.append(x)
            node_y.append(y)
            nodes.append((cid, ent))
            temp_node_pos[ent] = (x, y)
        node_pos.append(temp_node_pos)
    ents = list(set({ent for cid, ent in nodes}))
    ent_emb:np.ndarray = retriever.embed_paragraphs(ents, normalize=True, complete_return=False)
    q_emb:np.ndarray = retriever.embed_paragraphs(queries, normalize=True, complete_return=False)
    score_matrix = ent_emb @ q_emb.T
    score_matrix[score_matrix < retriever.syn_similarity] = 0
    ent2q = {ent : queries[scores.argmax()] for scores, ent in zip(score_matrix, ents) if scores.any()}
    node_text = [f'{cid}_{ent}: {ent2q[ent]}' if ent in ent2q else f'{cid}_{ent}' for cid, ent in nodes]
    node_color = [int(ent in ent2q) for cid, ent in nodes]
    node_symbol = [markers[queries.index(ent2q[ent]) + 1 if ent in ent2q else 0] for cid, ent in nodes]
            
    node_trace = go.Scatter(
        x=node_x, y=node_y, text=node_text,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_color,
            symbol=node_symbol,
            size=5,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=1
        )
    )
    
    edge_x = []
    edge_y = []
    edge_mid_x = []
    edge_mid_y = []
    edge_text = []
    for cid, ci in enumerate(results):
        for prev_id, relation_descriptions in ci.prev_relation_descriptions.items():
            prev_important_ents = results[cid + prev_id].important_ents
            prev_node_pos = node_pos[cid + prev_id]
            cur_important_ents = results[cid].important_ents
            cur_node_pos = node_pos[cid]
            for (ent1, ent2), relation_description in relation_descriptions:
                ent1 = ent1[0].upper() + ent1[1:]
                ent2 = ent2[0].upper() + ent2[1:]
                ent1 = ent1 if ent1 in prev_important_ents else sorted([ent for ent in prev_important_ents if ent.startswith(ent1)], key=lambda x: len(x))[0]
                ent2 = ent2 if ent2 in cur_important_ents else sorted([ent for ent in cur_important_ents if ent.startswith(ent2)], key=lambda x: len(x))[0]
                prev_x, prev_y = prev_node_pos[ent1]
                cur_x, cur_y = cur_node_pos[ent2]
                edge_x.extend([prev_x, cur_x, None])
                edge_y.extend([prev_y, cur_y, None])
                edge_mid_x.append((prev_x + cur_x) / 2)
                edge_mid_y.append((prev_y + cur_y) / 2)
                edge_text.append(relation_description)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        text=edge_text)
    
    sent_trace = go.Scatter(
        x=edge_mid_x, y=edge_mid_y, text=edge_text,
        mode='markers',
        marker_size=2,
        # textposition='top center',
        hoverinfo='text',
    )
    
    fig = go.Figure(data=[edge_trace, node_trace, sent_trace],
                layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()
    