import sys
import jsonlines
import json
from tqdm import tqdm
from sci_review.framework import *
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

command = sys.argv[1]
config_file = sys.argv[2]

with open(config_file) as f:
    config:dict = json.load(f)
    
dataset_dir = f'../../data/{config["dataset"]}'
dataset_file = f'{dataset_dir}/{config["dataset_file"]}'
question_types:list[str] = config["question_types"]
bad_ids:list[int] = config["bad_ids"]
is_temp:bool = config["is_temp"]
start_sid:int = config["start_sid"]
end_sid:int = config["end_sid"]
prefix:str = config["prefix"]
simple_load:bool = config["simple_load"]
llm_model = GPT_MODEL_CHEAP

# Retrieval config
retrieval2configs = {
    # RetrievalMethod.RAG: [
    #     {
    #         'k': 10
    #     }
    # ],
    # RetrievalMethod.GEN: [
    #     {
    #         'k': None
    #     }
    # ],
    # RetrievalMethod.CLS: [
    #     {
    #         'k': None
    #     }
    # ],
    RetrievalMethod.CITATION: [
        {
            'k': None
        }
    ],
}
    
with open('../../data/words_alpha.txt') as f:
    words_alpha = set(f.read().splitlines())
doc_manager = DocManager(word_vocab=words_alpha)

# Resource preparation
match command:
    case 'eval':
        if config['dataset'] == 'ACLSum':
            from script.retrieval.aclsum_base import *
            # run_eval = eval_aclsum
        else:
            raise ValueError(f'Unknown dataset: {config["dataset"]}')
        
        retrieval2configs[RetrievalMethod.RAG_BASE] = retrieval2configs[RetrievalMethod.RAG]
        eval_metrics = EvalMetrics()
        eval_results:dict[str, list[dict[str, float]]] = defaultdict(list[dict[str, float]])
    case 'inference':
        # Load Embeddings
        # embedding = HuggingFaceEmbeddings(
        #     model_name=DEFAULT_EMB_MODEL,
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': False}
        # )
        embedding = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device': 'cuda:0'},
            encode_kwargs={'normalize_embeddings': True}
        )
        results = defaultdict(list[Sample])

with jsonlines.open(dataset_file) as f_in:
    dataset = [Sample.model_validate(line) for line in f_in]
    if end_sid == -1:
        end_sid = len(dataset)
    for sid in tqdm(range(start_sid, end_sid)):
        print(sid)
        if sid in bad_ids:
            continue
        sample = dataset[sid]
        doc_manager.load_doc(doc_strs=sample.doc_blocks, outline=sample.outline, simple_load=simple_load)
        
        for retrieval_method, retrieval_configs in retrieval2configs.items():
            for retrieval_config in retrieval_configs:
                
                copied_sample = copy.deepcopy(sample)
                doc_manager.build_chunks(chunk_type=ChunkType.PARAGRAPH, max_seq_length=None, tokenizer=
                                         embedding._client.tokenizer if type(embedding) == HuggingFaceEmbeddings else embedding.client.tokenizer)
                if retrieval_method == RetrievalMethod.RAG:
                    vectorstore = Chroma.from_documents(documents=doc_manager.chunks, embedding=embedding)
                    retriever = vectorstore.as_retriever(search_kwargs={'k': retrieval_config['k']})
                    tool_list = [RetrieveByDenseRetrieval(retriever=retriever), RewriteQuestion]
                else:
                    tool_list = None
                
                # Run sample-specific inference/evaluation
                if command == 'inference':
                    for question_type in sample.question_types:
                        if not question_types or question_type in question_types:
                            run_framework(
                                doc_manager=doc_manager,
                                retrieval_method=retrieval_method, 
                                prefix=prefix, 
                                sid=sid, 
                                sample=copied_sample, 
                                question_type=question_type, 
                                is_temp=is_temp, 
                                data_dir=dataset_dir, 
                                llm_model=llm_model,
                                tool_list=tool_list,
                                **retrieval_config
                            )
                    results[f'{retrieval_method.value}_{retrieval_config["k"]}'].append(copied_sample)
                    
                if retrieval_method == RetrievalMethod.RAG:
                    vectorstore.delete_collection()
                    del vectorstore

                # elif command == 'eval':
                #     unique_ngram2sent = get_chunk_index([sent.text for section in doc_manager.sections if section.section_nlp_local for sent in section.section_nlp_local.sents])
                #     if load_from_pdf:
                #         valid_sent_ids = get_chunk_ids([sent for block in sample.doc_strs for sent in spacy_sent_tokenize(doc_manager.nlp, block)], unique_ngram2sent)
                #         if -1 in valid_sent_ids:
                #             print(f'Invalid sent id in sample {sid}, retrieval_config {retrieval_config}, {valid_sent_ids.count(-1)}/{len(valid_sent_ids)}')
                #             valid_sent_ids = [sent_id for sent_id in valid_sent_ids if sent_id > -1]
                #         valid_sent_ids = set(valid_sent_ids)
                #     else:
                #         valid_sent_ids = set(range(max(sent_id for ngram, (sent_id, sent) in unique_ngram2sent.items()) + 1))
                
                #     for question_type in question_types:
                #         eval_result = run_eval(
                #             doc_manager=doc_manager, 
                #             eval_metrics=eval_metrics, 
                #             valid_sent_ids=valid_sent_ids, 
                #             unique_ngram2sent=unique_ngram2sent, 
                #             retrieval_method=retrieval_method, 
                #             prefix=prefix, 
                #             sid=sid, 
                #             sample=sample, 
                #             load_from_pdf=load_from_pdf, 
                #             question_type=question_type, 
                #             is_temp=is_temp,
                #             **retrieval_config
                #         )
                #         eval_file = get_eval_file(retrieval_method=retrieval_method, prefix=prefix, question_type=question_type, load_from_pdf=load_from_pdf, is_temp=is_temp, data_dir=dataset_dir, **retrieval_config)
                #         eval_results[eval_file].append(eval_result)
                                        
# Post-process
if sys.argv[1] == 'inference':
    for retrieval_setting, samples in results.items():
        with jsonlines.open(f'{dataset_dir}/{prefix}_{retrieval_setting}_{dataset_file.rsplit("/")[-1].rsplit(".", maxsplit=1)[0]}.json', 'w') as f_out:
            f_out.write_all([sample.model_dump() for sample in samples])
            
# elif sys.argv[1] == 'eval':
#     for eval_file, eval_result in eval_results.items():
#         dir = os.path.dirname(eval_file)
#         if not os.path.exists(dir):
#             os.makedirs(dir)
#         with open(eval_file, 'w') as f_out:
#             json.dump(eval_result, f_out)
#         retrieval_method, prefix, question_type, load_from_pdf, sent_chunk, max_seq_len, k, is_temp = parse_eval_file(eval_file, dataset_dir)
#         print(retrieval_method, f'prefix--{prefix}', f'question_type--{question_type}', f'load_from_pdf--{load_from_pdf}', f'sent_chunk--{sent_chunk}', f'max_seq_len--{max_seq_len}', f'k--{k}')
#         print('recall', np.mean([result['recall'] for result in eval_result[:]]))
#         print('precision', np.mean([result['precision'] for result in eval_result[:]]))
#         print('f1', np.mean([result['f1'] for result in eval_result[:]]))
#         print('')
