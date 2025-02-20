import sys
import jsonlines
import json
from tqdm import tqdm
from sci_review.framework import *

command = sys.argv[1]
config_file = sys.argv[2]

with open(config_file) as f:
    config:dict = json.load(f)
    
dataset_dir = f'../../data/{config["dataset"]}'
dataset_file = f'{dataset_dir}/{config["dataset_file"]}'
bad_sids:list[int] = config["bad_sids"]
question_types:list[str] = config["question_types"]
load_from_pdf:bool = config["load_from_pdf"]
is_temp:bool = config["is_temp"]
start_sid:int = config["start_sid"]
end_sid:int = config["end_sid"]
prefix:str = config["prefix"]
simple_load:bool = config["simple_load"]
    
with open('../../data/words_alpha.txt') as f:
    words_alpha = set(f.read().splitlines())
doc_manager = DocManager(word_vocab=words_alpha)

# Resource preparation
if command == 'inference':
    agentic_rag = AgenticRAG(llm_model=GPT_MODEL_EXPENSIVE)
    agentic_rag.doc_manager = doc_manager
elif command == 'eval':
    if config['dataset'] == 'ACLSum':
        from script.retrieval.aclsum_base import *
        run_eval = eval_aclsum
    else:
        raise ValueError(f'Unknown dataset: {config["dataset"]}')
    
    retrieval2configs[RAG_BASE] = retrieval2configs[RAG]
    eval_metrics = EvalMetrics()
    eval_results:dict[str, list[dict[str, float]]] = defaultdict(list[dict[str, float]])
else:
    raise ValueError(f'Unknown command: {command}')

with jsonlines.open(dataset_file) as f_in:
    dataset = [Sample.model_validate(line) for line in f_in]
    for sid in tqdm(range(start_sid, end_sid)):
        print(sid)
        if sid in bad_sids:
            continue
        sample = dataset[sid]
        
        load_doc_manager(doc_manager, sample, load_from_pdf, dataset_dir, simple_load)
        for retrieval_method, retrieval_configs in retrieval2configs.items():
            for retrieval_config in retrieval_configs:
                doc_manager.build_chunks(sent_chunk=retrieval_config['sent_chunk'], max_seq_length=retrieval_config['max_seq_len'])
                
                # Run sample-specific inference/evaluation
                if command == 'inference':
                    for question_type in sample.question_types:
                        if not question_types or question_type in question_types:
                            run_framework(
                                agentic_rag=agentic_rag, 
                                retrieval_method=retrieval_method, 
                                prefix=prefix, 
                                sid=sid, 
                                sample=sample, 
                                load_from_pdf=load_from_pdf, 
                                question_type=question_type, 
                                is_temp=is_temp, 
                                data_dir=dataset_dir, 
                                **retrieval_config
                            )

                elif command == 'eval':
                    unique_ngram2sent = get_sent_index([sent.text for section in doc_manager.sections if section.section_nlp_local for sent in section.section_nlp_local.sents])
                    if load_from_pdf:
                        valid_sent_ids = get_sent_ids([sent for block in sample.doc_strs for sent in spacy_sent_tokenize(doc_manager.nlp, block)], unique_ngram2sent)
                        if -1 in valid_sent_ids:
                            print(f'Invalid sent id in sample {sid}, retrieval_config {retrieval_config}, {valid_sent_ids.count(-1)}/{len(valid_sent_ids)}')
                            valid_sent_ids = [sent_id for sent_id in valid_sent_ids if sent_id > -1]
                        valid_sent_ids = set(valid_sent_ids)
                    else:
                        valid_sent_ids = set(range(max(sent_id for ngram, (sent_id, sent) in unique_ngram2sent.items()) + 1))
                
                    for question_type in question_types:
                        eval_result = run_eval(
                            doc_manager=doc_manager, 
                            eval_metrics=eval_metrics, 
                            valid_sent_ids=valid_sent_ids, 
                            unique_ngram2sent=unique_ngram2sent, 
                            retrieval_method=retrieval_method, 
                            prefix=prefix, 
                            sid=sid, 
                            sample=sample, 
                            load_from_pdf=load_from_pdf, 
                            question_type=question_type, 
                            is_temp=is_temp,
                            **retrieval_config
                        )
                        eval_file = get_eval_file(retrieval_method=retrieval_method, prefix=prefix, question_type=question_type, load_from_pdf=load_from_pdf, is_temp=is_temp, data_dir=dataset_dir, **retrieval_config)
                        eval_results[eval_file].append(eval_result)
                                        
# Post-process
if sys.argv[1] == 'eval':
    for eval_file, eval_result in eval_results.items():
        dir = os.path.dirname(eval_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(eval_file, 'w') as f_out:
            json.dump(eval_result, f_out)
        retrieval_method, prefix, question_type, load_from_pdf, sent_chunk, max_seq_len, k, is_temp = parse_eval_file(eval_file, dataset_dir)
        print(retrieval_method, f'prefix--{prefix}', f'question_type--{question_type}', f'load_from_pdf--{load_from_pdf}', f'sent_chunk--{sent_chunk}', f'max_seq_len--{max_seq_len}', f'k--{k}')
        print('recall', np.mean([result['recall'] for result in eval_result[:]]))
        print('precision', np.mean([result['precision'] for result in eval_result[:]]))
        print('f1', np.mean([result['f1'] for result in eval_result[:]]))
        print('')
