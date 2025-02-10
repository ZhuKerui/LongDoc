from sci_review.data_base import *
import jsonlines
from sci_review.agentic_rag import *
from tqdm import tqdm
from typing import Any


ACLSUM_DIR = '../../data/ACLSum'
ACLSUM_PDF_DIR = f'{ACLSUM_DIR}/pdf'
ACLSUM_GENERATION_DIR = f'{ACLSUM_DIR}/generation'
ACLSUM_TEMP_DIR = f'{ACLSUM_DIR}/temp'
ACLSUM_EVALUATION_DIR = f'{ACLSUM_DIR}/evaluation'

bad_pdfs = ['2021.eacl-main.251.pdf']

def load_doc_manager(doc_manager:DocManager, sample:Sample, load_from_pdf:bool):
    if load_from_pdf:
        # Load from full pdf
        doc_file = f"{ACLSUM_PDF_DIR}/{sample.doc_file.split('/')[-1]}"
        outline_file = f"{ACLSUM_PDF_DIR}/outline_{sample.doc_file.split('/')[-1].replace('.pdf', '.txt')}"
        if not os.path.exists(doc_file):
            download_file(sample.doc_file, doc_file)
        if os.path.exists(outline_file):
            with open(outline_file) as f:
                outline = f.read()
        else:
            outline = None
        doc_manager.load_doc(doc_file=doc_file, outline=outline)
        if not outline:
            with open(outline_file, 'w') as f:
                f.write(doc_manager.full_outline)
    else:
        # Load from partial text
        doc_manager.load_doc(doc_strs=sample.doc_strs, outline=sample.outline)
        
def get_process_file(retrieval_method:str, split:str, sid:int, question_type:str, load_from_pdf:bool, sent_chunk:bool, max_seq_len:int, k:int = None, is_temp:bool = False):
    generation_dir = ACLSUM_TEMP_DIR if is_temp else ACLSUM_GENERATION_DIR
    if retrieval_method == 'rag':
        return f'{generation_dir}/{retrieval_method}/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json'
    elif retrieval_method == 'gen':
        return f'{generation_dir}/{retrieval_method}/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}.json'
    elif retrieval_method == 'rag_base':
        return f'{generation_dir}/rag/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json'
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    
def get_eval_file(retrieval_method:str, split:str, question_type:str, load_from_pdf:bool, sent_chunk:bool, max_seq_len:int, k:int = None, is_temp:bool = False):
    evaluation_dir = ACLSUM_TEMP_DIR if is_temp else ACLSUM_EVALUATION_DIR
    if retrieval_method in {'rag', 'rag_base'}:
        return f"{evaluation_dir}/{retrieval_method}/{split}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json"
    elif retrieval_method == 'gen':
        return f"{evaluation_dir}/{retrieval_method}/{split}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}.json"
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
        
def run_aclsum(
    agentic_rag:AgenticRAG,
    retrieval_method:str,
    split:str, 
    sid:int, 
    sample:Sample, 
    load_from_pdf:bool, 
    question_type:str, 
    sent_chunk:bool, 
    max_seq_len:int, 
    k:int,
    is_temp:bool = False
    ):
    
    process_file = get_process_file(retrieval_method, split, sid, question_type, load_from_pdf, sent_chunk, max_seq_len, k, is_temp)
    if retrieval_method == 'rag':
        agentic_rag.load_langgraph([RetrieveByDenseRetrieval(agentic_rag.doc_manager, k), RewriteQuestion])
        process = agentic_rag.invoke(sample.questions[question_type])
    elif retrieval_method == 'gen':
        agentic_rag.load_langgraph()
        content = PARAGRAPH_SEP.join([f'Chunk {chunk.metadata["chunk_id"]}: {chunk.page_content}' for chunk in agentic_rag.doc_manager.chunks])
        prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nSelect the Chunk ids that are relevant to the following question: \n\n{sample.questions[question_type]}\n\nReturn only the selected chunk ids separated by commas, e.g. "1, 3, 5".'
        process = agentic_rag.invoke(prompt)
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    
    AgenticRAG.dump_process(process, process_file)
        
def get_sents_and_process(
    doc_manager:DocManager,
    retrieval_method:str,
    split:str,
    sid:int,
    question_type:str,
    load_from_pdf:bool,
    sent_chunk:bool,
    max_seq_len:int,
    k:int = None,
    is_temp:bool = False
):
    process_file = get_process_file(retrieval_method, split, sid, question_type, load_from_pdf, sent_chunk, max_seq_len, k, is_temp)
    process = AgenticRAG.load_process(process_file)
    
    if retrieval_method == 'rag':
        passages = list[str]()
        for step in process[::-1]:
            if AgenticRAG.TOOL_POST_PROCESS in step and step[AgenticRAG.TOOL_POST_PROCESS]:
                passages.extend(step[AgenticRAG.TOOL_POST_PROCESS][AgenticRAG.PASSAGES])
                break
        
    elif retrieval_method == 'gen':
        chunk_ids = list(map(int, process[0][AgenticRAG.AGENT][AgenticRAG.MESSAGES][0].content.split(', ')))
        passages = [doc_manager.chunks[chunk_id].page_content for chunk_id in chunk_ids]
        
    elif retrieval_method == 'rag_base':
        passages = list[str]()
        for step in process:
            if AgenticRAG.TOOL_POST_PROCESS in step and step[AgenticRAG.TOOL_POST_PROCESS]:
                passages.extend(passage.split(CHUNK_SEP)[1] for passage in step[AgenticRAG.TOOL_POST_PROCESS][AgenticRAG.RETRIEVAL])
        passages = list(set(passages))
    
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    
    retrieved_sents = passages if sent_chunk else [sent for passage in passages for sent in spacy_sent_tokenize(doc_manager.nlp, passage)]
    return process, retrieved_sents
    
def eval_aclsum(
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
        is_temp=is_temp)
    if not retrieved_sents:
        return {'sid': sid, 'precision': 0, 'recall': 0, 'f1': 0}
    retrieved_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_sent_ids(retrieved_sents, unique_ngram2sent))]
    gold_sent_ids = [sent_label if sent_id in valid_sent_ids else 0 for sent_id, sent_label in enumerate(get_binary_sent_ids(sample.extractions[question_type], unique_ngram2sent))]
    
    eval_result:dict[str, Any] = eval_metrics.eval_precision_recall_f1(predictions=retrieved_sent_ids, references=gold_sent_ids)
    eval_result.update({'sid': sid})
    return eval_result
        
if __name__ == '__main__':
    
    import sys
    
    # Dataset config
    split = 'train'
    load_from_pdf = False
    # load_from_pdf = True
    
    # Retrieval config
    retrieval2configs = {
        'rag': [
            {
                'sent_chunk': True, 
                'max_seq_len': None, 
                'k': 10
            },
            {
                'sent_chunk': False, 
                'max_seq_len': 100, 
                'k': 10
            },
            # {
            #     'sent_chunk': False, 
            #     'max_seq_len': None, 
            #     'k': 3
            # }
        ],
        # 'gen': [
        #     {
        #         'sent_chunk': True, 
        #         'max_seq_len': None, 
        #         'k': None
        #     }
        # ]
    }
    
    with open('words_alpha.txt') as f:
        words_alpha = set(f.read().splitlines())
    doc_manager = DocManager(word_vocab=words_alpha)
    
    # Resource preparation
    if sys.argv[1] == 'inference':
        agentic_rag = AgenticRAG()
        agentic_rag.doc_manager = doc_manager
    elif sys.argv[1] == 'eval':
        retrieval2configs['rag_base'] = retrieval2configs['rag']
        eval_metrics = EvalMetrics()
        eval_results:dict[str, list[dict[str, float]]] = defaultdict(list[dict[str, float]])
    else:
        raise ValueError(f'Unknown command: {sys.argv[1]}')

    with jsonlines.open(f'{ACLSUM_DIR}/{split}_dataset.jsonl') as f_in:
        aclsum_dataset = [Sample.model_validate(line) for line in f_in]
        for sid in tqdm(range(0, 100)):
            print(sid)
            if sid in [60, 70, 76]:
                continue
            sample = aclsum_dataset[sid]
            if sample.doc_file.split('/')[-1] in bad_pdfs:
                continue
                
            load_doc_manager(doc_manager, sample, load_from_pdf)
            for retrieval_method, retrieval_configs in retrieval2configs.items():
                for retrieval_config in retrieval_configs:
                    doc_manager.build_chunks(sent_chunk=retrieval_config['sent_chunk'], max_seq_length=retrieval_config['max_seq_len'])
                    
                    # Run sample-specific inference/evaluation
                    if sys.argv[1] == 'inference':
                        for question_type in ['challenge', 'approach', 'outcome']:
                            run_aclsum(
                                agentic_rag=agentic_rag, 
                                retrieval_method=retrieval_method, 
                                split=split, 
                                sid=sid, 
                                sample=sample, 
                                load_from_pdf=load_from_pdf, 
                                question_type=question_type, 
                                is_temp=True, 
                                **retrieval_config
                            )

                    elif sys.argv[1] == 'eval':
                        unique_ngram2sent = get_sent_index([sent.text for section in doc_manager.sections if section.section_nlp_local for sent in section.section_nlp_local.sents])
                        if load_from_pdf:
                            valid_sent_ids = get_sent_ids([sent for block in sample.doc_strs if block not in ['Abstract', 'Introduction', 'Conclusion'] for sent in spacy_sent_tokenize(doc_manager.nlp, block)], unique_ngram2sent)
                            if -1 in valid_sent_ids:
                                print(f'Invalid sent id in sample {sid}, retrieval_config {retrieval_config}, {valid_sent_ids.count(-1)}/{len(valid_sent_ids)}')
                                valid_sent_ids = [sent_id for sent_id in valid_sent_ids if sent_id > -1]
                            valid_sent_ids = set(valid_sent_ids)
                        else:
                            valid_sent_ids = set(range(max(sent_id for ngram, (sent_id, sent) in unique_ngram2sent.items()) + 1))
                    
                        for question_type in ['challenge', 'approach', 'outcome']:
                            eval_result = eval_aclsum(
                                doc_manager=doc_manager, 
                                eval_metrics=eval_metrics, 
                                valid_sent_ids=valid_sent_ids, 
                                unique_ngram2sent=unique_ngram2sent, 
                                retrieval_method=retrieval_method, 
                                split=split, 
                                sid=sid, 
                                sample=sample, 
                                load_from_pdf=load_from_pdf, 
                                question_type=question_type, 
                                is_temp=True,
                                **retrieval_config
                            )
                            eval_file = get_eval_file(retrieval_method=retrieval_method, split=split, question_type=question_type, load_from_pdf=load_from_pdf, is_temp=True, **retrieval_config)
                            eval_results[eval_file].append(eval_result)
                            
    # Post-process
    if sys.argv[1] == 'eval':
        for eval_file, eval_result in eval_results.items():
            dir = os.path.dirname(eval_file)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(eval_file, 'w') as f_out:
                json.dump(eval_result, f_out)
