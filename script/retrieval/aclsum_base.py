from sci_review.framework import *
import jsonlines
from typing import Any


ACLSUM_DIR = '../../data/ACLSum'

challenge_question = 'Summarize the challenge of the paper, which is the current situation faced by the researcher.'
approach_question = 'Summarize the approach of the paper: How they intend to carry out the investigation, comments on a theoretical model or framework.'
outcome_question = 'Summarize the outcome of the paper: Overall conclusion that should reject or support the research hypothesis.'

question_type2question = {
    'challenge': challenge_question,
    'approach': approach_question,
    'outcome': outcome_question
}

# def eval_aclsum(
#     doc_manager:DocManager,
#     eval_metrics:EvalMetrics,
#     valid_sent_ids:set[int],
#     unique_ngram2sent:dict[tuple, tuple[int, str]],
#     retrieval_method:str,
#     prefix:str, 
#     sid:int, 
#     sample:Sample, 
#     load_from_pdf:bool, 
#     question_type:str, 
#     sent_chunk:bool, 
#     max_seq_len:int, 
#     k:int = None, 
#     is_temp=False
#     ):
#     _, retrieved_sents = get_sents_and_process(
#         doc_manager=doc_manager,
#         retrieval_method=retrieval_method,
#         prefix=prefix,
#         sid=sid,
#         question_type=question_type,
#         load_from_pdf=load_from_pdf,
#         sent_chunk=sent_chunk,
#         max_seq_len=max_seq_len,
#         k=k,
#         is_temp=is_temp,
#         data_dir=ACLSUM_DIR)
#     if not retrieved_sents:
#         return {'sid': sid, 'precision': 0, 'recall': 0, 'f1': 0}
#     retrieved_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_chunk_ids(retrieved_sents, unique_ngram2sent))]
#     gold_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_chunk_ids(sample.extractions[question_type], unique_ngram2sent))]
    
#     eval_result:dict[str, Any] = eval_metrics.eval_precision_recall_f1(predictions=retrieved_sent_ids, references=gold_sent_ids)
#     eval_result.update({'sid': sid})
#     return eval_result
        
if __name__ == '__main__':
    
    import sys
    from tqdm import tqdm
    
    # Dataset config
    split = 'train'
    prefix = split
    
    if sys.argv[1] == 'build_data':
        from aclsum import ACLSum
        train = ACLSum(split)
        
        with open('../../data/words_alpha.txt') as f:
            words_alpha = set(f.read().splitlines())
        doc_manager = DocManager(word_vocab=words_alpha)

        aclsum_dataset = list[Sample]()
        pdf_path = pdf_dir(ACLSUM_DIR)
        if not os.path.exists(pdf_path):
            os.makedirs(pdf_path)
            
        for doc in tqdm(train):
            paper_ids = get_paper_ids_by_semantic_scholar(acl_id=doc.id)
            arxiv_id = paper_ids[ARXIV]
            doc_file = f"{ARXIV}:{arxiv_id}" if arxiv_id is not None else f"{ACL}:{doc.id}"
            file_path = f"{pdf_path}/{doc_file}.pdf"
            outline = ''
            if arxiv_id is not None:
                paragraphs, outline, title = get_arxiv_paper_text(arxiv_id)
            if outline:
                download_arxiv_pdf(arxiv_id, file_path)
                paragraphs = [DocManager.remove_citations(p) for p in paragraphs]
            else:
                download_acl_pdf(doc.id, file_path)
                try:
                    blocks, sections, meta_data, outline, main_text_style, doc_vocab = doc_manager.parse_pdf(file_path)
                except Exception as e:
                    print(f'Error in parsing {file_path}')
                    continue
                paragraphs = [block.text for block in blocks]
                
            unique_ngram2chunk = get_chunk_index(paragraphs)
            labeled_texts = [
                DocManager.remove_citations(DocManager.remove_space_before_punct(' '.join(doc.get_all_sentences(['abstract'])))), 
                DocManager.remove_citations(DocManager.remove_space_before_punct(' '.join(doc.get_all_sentences(['introduction'])))), 
                DocManager.remove_citations(DocManager.remove_space_before_punct(' '.join(doc.get_all_sentences(['conclusion']))))
            ]
            
            doc_blocks_with_label = [False] * len(paragraphs)
            for chunk_ids in get_chunk_ids(labeled_texts, unique_ngram2chunk):
                for chunk_id in chunk_ids:
                    doc_blocks_with_label[chunk_id] = True
            
            question_types = list(question_type2question.keys())
            questions = dict[str, list[str]]()
            relevant_blocks = dict[str, list[int]]()
            extractions = dict[str, list[tuple[str, list[int]]]]()
            answers = dict[str, str]()
            for question_type, question in question_type2question.items():
                questions[question_type] = [question]
                relevant_blocks[question] = []
                extractions[question] = []
                answers[question] = doc.summaries[question_type]
                sents = [DocManager.remove_citations(DocManager.remove_space_before_punct(sent)) for sent in doc.get_all_highlighted_sentences(question_type)]
                for chunk_ids, sent in zip(get_chunk_ids(sents, unique_ngram2chunk), sents):
                    chunk_ids = [chunk_id for chunk_id in chunk_ids if doc_blocks_with_label[chunk_id]]
                    if chunk_ids:
                        relevant_blocks[question].extend(chunk_ids)
                        extractions[question].append((sent, chunk_ids))
                relevant_blocks[question] = list(set(relevant_blocks[question]))
            
            aclsum_dataset.append(Sample(
                # Doc info
                doc_file=doc_file,
                doc_blocks=paragraphs,
                outline=outline,
                doc_blocks_with_label=doc_blocks_with_label,
                
                # Question info
                question_types=question_types,
                questions=questions,
                
                # Answer info
                relevant_blocks=relevant_blocks,
                answers=answers,
                extractions=extractions
            ))
            
        save_dataset_to_jsonl(aclsum_dataset, f'{ACLSUM_DIR}/{split}_dataset.jsonl')
            