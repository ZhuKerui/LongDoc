import sys
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
llm_model = GPT_MODEL_EXPENSIVE

# Retrieval config
retrieval2configs = {
    RetrievalMethod.RAG: [
        {
            'k': 10
        }
    ],
    RetrievalMethod.GEN: [
        {
            'k': None
        }
    ],
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
    RetrievalMethod.EXPLANATION: [
        {
            'k': None
        }
    ],
}

match command:
    case 'inference':
        with open('../../data/words_alpha.txt') as f:
            words_alpha = set(f.read().splitlines())
        doc_manager = DocManager(word_vocab=words_alpha)
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
        dataset = load_dataset_from_jsonl(dataset_file)
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
                    
                    for question_type in sample.question_types:
                        if not question_types or question_type in question_types:
                            run_framework(
                                doc_manager=doc_manager,
                                retrieval_method=retrieval_method, 
                                prefix=f'{prefix}_{dataset_file.rsplit("/")[-1].rsplit(".", maxsplit=1)[0]}', 
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
    
        for retrieval_setting, samples in results.items():
            save_dataset_to_jsonl(samples, f'{dataset_dir}/{prefix}_{dataset_file.rsplit("/")[-1].rsplit(".", maxsplit=1)[0]}_{retrieval_setting}.jsonl')
                
    case 'eval':
        eval_by_qtype:bool = config['eval_by_qtype']
        # retrieval2configs[RetrievalMethod.RAG_BASE] = retrieval2configs[RetrievalMethod.RAG]
        eval_metrics = EvalMetrics()
        
        for retrieval_method, retrieval_configs in retrieval2configs.items():
            for retrieval_config in retrieval_configs:
                retrieval_setting = f'{retrieval_method.value}_{retrieval_config["k"]}'
                dataset = load_dataset_from_jsonl(f'{dataset_dir}/{prefix}_{dataset_file.rsplit("/")[-1].rsplit(".", maxsplit=1)[0]}_{retrieval_setting}.jsonl')
                eval_selected_blocks(dataset=dataset, eval_metrics=eval_metrics)
                
                print(retrieval_setting)
                print_eval_selected_blocks(dataset=dataset, eval_by_qtype=eval_by_qtype)
                        
                save_dataset_to_jsonl(dataset, f'{dataset_dir}/{prefix}_{dataset_file.rsplit("/")[-1].rsplit(".", maxsplit=1)[0]}_{retrieval_setting}_eval.jsonl')
