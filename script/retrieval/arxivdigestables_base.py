from sci_review.framework import *

import jsonlines
from typing import Any
from pymongo import MongoClient
import pymongo

# Making Connection
myclient = MongoClient("mongodb://localhost:27017/") 
# database 
db = myclient["ArxivDIGESTables"]

full_texts_collection = db["full_texts"]
papers_collection = db["papers"]
tables_collection = db["tables"]

ARXIVDIGEST_DIR = '../../data/ArxivDIGESTables'
ARXIVDIGEST_CORPUS2DOCFILE = f'{ARXIVDIGEST_DIR}/corpus2docfile.json'

bad_pdfs = ['2021.eacl-main.251.pdf']
    
def eval_arxivdigest(
    doc_manager:DocManager,
    eval_metrics:EvalMetrics,
    valid_sent_ids:set[int],
    unique_ngram2sent:dict[tuple, tuple[int, str]],
    retrieval_method:str,
    split:str, 
    sid:int, 
    sample:Sample, 
    load_from_pdf:bool, 
    question_type:str, 
    sent_chunk:bool, 
    max_seq_len:int, 
    k:int = None, 
    is_temp=False
    ):
    _, retrieved_sents = get_sents_and_process(
        doc_manager=doc_manager,
        retrieval_method=retrieval_method,
        split=split,
        sid=sid,
        question_type=question_type,
        load_from_pdf=load_from_pdf,
        sent_chunk=sent_chunk,
        max_seq_len=max_seq_len,
        k=k,
        is_temp=is_temp,
        data_dir=ARXIVDIGEST_DIR
    )
    if not retrieved_sents:
        return {'sid': sid, 'precision': 0, 'recall': 0, 'f1': 0}
    retrieved_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_sent_ids(retrieved_sents, unique_ngram2sent))]
    gold_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_sent_ids(sample.extractions[question_type], unique_ngram2sent))]
    
    eval_result:dict[str, Any] = eval_metrics.eval_precision_recall_f1(predictions=retrieved_sent_ids, references=gold_sent_ids)
    eval_result.update({'sid': sid})
    return eval_result
        
def generate_question_description(doc_manager:DocManager, column:str, caption:str, in_text_ref:str=None):
    if in_text_ref is None:
        CAPTION_PROMPT = f"""A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. This table contains a column called {column}. Please write a brief definition for this column.
        
        Here is the caption for the table: {caption}.
        
        Definition: """.replace('    ', '')
    else:
        CAPTION_PROMPT = f"""A user is making a table for a scholarly paper that contains information about multiple papers and compares these papers. This table contains a column called {column}. Please write a brief definition for this column.
        
        Here is the caption for the table: {caption}.
        Following is some additional information about this table: {in_text_ref}.
        
        Definition: """.replace('    ', '')
    return doc_manager.client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': CAPTION_PROMPT,
            }
        ],
        model=doc_manager.tool_llm,
    ).choices[0].message.content.strip()
    
def generate_question(doc_manager:DocManager, question_description:str):
    CONTEXT_QUERY = f"""Rewrite this description as a one-line question.
    
    Question description: {question_description}
    
    Question: """.replace('    ', '')
    
    return doc_manager.client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': CONTEXT_QUERY,
            }
        ],
        model=doc_manager.tool_llm,
    ).choices[0].message.content.strip()
    
if __name__ == '__main__':
    
    import sys
    
    if sys.argv[1] == 'build_data':
        
        doc_manager = DocManager()

        arxivdigest_dataset = list[Sample]()
        if os.path.exists(ARXIVDIGEST_CORPUS2DOCFILE):
            with open(ARXIVDIGEST_CORPUS2DOCFILE, 'r') as f_in:
                corpus_id2doc_file:dict[str, str] = json.load(f_in)
        else:
            corpus_id2doc_file = dict[str, str]()
            
        for tid, table in enumerate(tables_collection.find().sort('tabid', pymongo.ASCENDING).limit(10)):
            print(f"Processing table {tid}")
            corpus_id2sample = dict[str, Sample]()
            for bib in table['row_bib_map']:
                corpus_id = bib['corpus_id']
                corpus_id_str = str(corpus_id)
                # if corpus_id not in corpus_id2doc_file:
                paper_ids = get_paper_ids_by_semantic_scholar(corpus_id=corpus_id)
                arxiv_id, acl_id = paper_ids[ARXIV], paper_ids[ACL]
                if arxiv_id is not None:
                    doc_file = f"{ARXIV}:{arxiv_id}"
                elif acl_id is not None:
                    doc_file = f"{ACL}:{acl_id}"
                else:
                    continue
                corpus_id2doc_file[corpus_id_str] = doc_file
                corpus_id2sample[corpus_id_str] = Sample(doc_file=doc_file)
            if corpus_id2sample:
                in_text_ref = []
                for ref in table['in_text_ref']:
                    ref_text = ref['text'].replace('\n', ' ').replace(table['tabid'], 'TARGET_TABLE')
                    if ref_text not in in_text_ref:
                        in_text_ref.append(ref_text)
                in_text_ref = '\n\n'.join(in_text_ref)
                for column, paper_values in table['table'].items():
                    question_description = generate_question_description(doc_manager, column, table['caption'], in_text_ref)
                    question = generate_question(doc_manager, question_description)
                    for corpus_id_str, values in paper_values.items():
                        sample = corpus_id2sample.get(corpus_id_str, None)
                        if sample is None:
                            continue
                        sample.question_types.append(column)
                        sample.question_types.sort()
                        sample.questions[column] = [question]
                        sample.question_meta[question] = {
                            'description': question_description,
                            'in_text_ref': in_text_ref,
                            'column': column,
                            'caption': table['caption'],
                            'table_id': table['tabid'],
                            'corpus_id': corpus_id_str
                        }
                        sample.answers[question] = values[0]
                arxivdigest_dataset.extend(corpus_id2sample.values())
                
        arxivdigest_dataset.sort(key=lambda x: (x.doc_file, x.question_types))
                
        with open(ARXIVDIGEST_CORPUS2DOCFILE, 'w') as f_out:
            json.dump(corpus_id2doc_file, f_out)
            
        with jsonlines.open(f'{ARXIVDIGEST_DIR}/dataset.jsonl', 'w') as f_out:
            f_out.write_all([sample.model_dump() for sample in arxivdigest_dataset])
            
    
    else:
        raise ValueError(f'Unknown command: {sys.argv[1]}')