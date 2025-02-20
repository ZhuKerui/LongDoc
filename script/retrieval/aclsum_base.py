from sci_review.framework import *
import jsonlines
from typing import Any


ACLSUM_DIR = '../../data/ACLSum'

challenge_question = 'Summarize the challenge of the paper, which is the current situation faced by the researcher.'
approach_question = 'Summarize the approach of the paper: How they intend to carry out the investigation, comments on a theoretical model or framework.'
outcome_question = 'Summarize the outcome of the paper: Overall conclusion that should reject or support the research hypothesis.'

def eval_aclsum(
    doc_manager:DocManager,
    eval_metrics:EvalMetrics,
    valid_sent_ids:set[int],
    unique_ngram2sent:dict[tuple, tuple[int, str]],
    retrieval_method:str,
    prefix:str, 
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
        prefix=prefix,
        sid=sid,
        question_type=question_type,
        load_from_pdf=load_from_pdf,
        sent_chunk=sent_chunk,
        max_seq_len=max_seq_len,
        k=k,
        is_temp=is_temp,
        data_dir=ACLSUM_DIR)
    if not retrieved_sents:
        return {'sid': sid, 'precision': 0, 'recall': 0, 'f1': 0}
    retrieved_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_sent_ids(retrieved_sents, unique_ngram2sent))]
    gold_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_sent_ids(sample.extractions[question_type], unique_ngram2sent))]
    
    eval_result:dict[str, Any] = eval_metrics.eval_precision_recall_f1(predictions=retrieved_sent_ids, references=gold_sent_ids)
    eval_result.update({'sid': sid})
    return eval_result
        
if __name__ == '__main__':
    
    import sys
    from tqdm import tqdm
    
    # Dataset config
    split = 'train'
    prefix = split
    
    if sys.argv[1] == 'build_data':
        # Load per split ("train", "val", "test")
        from aclsum import ACLSum
        train = ACLSum(split)

        aclsum_dataset = list[dict]()
        for doc in tqdm(train):
            paper_ids = get_paper_ids_by_semantic_scholar(acl_id=doc.id)
            arxiv_id = paper_ids[ARXIV]
            if arxiv_id is not None:
                doc_file = f"{ARXIV}:{arxiv_id}"
            else:
                doc_file = f"{ACL}:{doc.id}"
            aclsum_dataset.append(Sample(
                doc_file=doc_file,
                doc_strs=[
                    'Abstract', 
                    DocManager.remove_citations(DocManager.remove_space_before_punct(' '.join(doc.get_all_sentences(['abstract'])))), 
                    'Introduction', 
                    DocManager.remove_citations(DocManager.remove_space_before_punct(' '.join(doc.get_all_sentences(['introduction'])))), 
                    'Conclusion', 
                    DocManager.remove_citations(DocManager.remove_space_before_punct(' '.join(doc.get_all_sentences(['conclusion'])))), 
                ],
                outline='Abstract\nIntroduction\nConclusion',
                question_types=['challenge', 'approach', 'outcome'],
                questions={
                    'challenge': [challenge_question], 
                    'approach': [approach_question], 
                    'outcome': [outcome_question]
                },
                answers={
                    challenge_question: doc.summaries['challenge'], 
                    approach_question: doc.summaries['approach'], 
                    outcome_question: doc.summaries['outcome']
                },
                extractions={
                    challenge_question: [DocManager.remove_citations(DocManager.remove_space_before_punct(sent)) for sent in doc.get_all_highlighted_sentences('challenge')],
                    approach_question: [DocManager.remove_citations(DocManager.remove_space_before_punct(sent)) for sent in doc.get_all_highlighted_sentences('approach')],
                    outcome_question: [DocManager.remove_citations(DocManager.remove_space_before_punct(sent)) for sent in doc.get_all_highlighted_sentences('outcome')],
                }
            ).model_dump())
            
        with jsonlines.open(f'{ACLSUM_DIR}/{split}_dataset.jsonl', 'w') as f_out:
            f_out.write_all(aclsum_dataset)
            