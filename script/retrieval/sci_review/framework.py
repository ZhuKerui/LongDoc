
from .agentic_rag import *
from .data_base import *

RAG = 'rag'
GEN = 'gen'
RAG_BASE = 'rag_base'
CLS_GEN = 'cls_gen'

# Retrieval config
retrieval2configs = {
    RAG: [
        # {
        #     'sent_chunk': True, 
        #     'max_seq_len': None, 
        #     'k': 10
        # },
        # {
        #     'sent_chunk': False, 
        #     'max_seq_len': 100, 
        #     'k': 10
        # },
        # {
        #     'sent_chunk': False, 
        #     'max_seq_len': None, 
        #     'k': 3
        # }
    ],
    GEN: [
        # {
        #     'sent_chunk': True, 
        #     'max_seq_len': None, 
        #     'k': None
        # }
    ],
    CLS_GEN: [
        {
            'sent_chunk': True, 
            'max_seq_len': None, 
            'k': None
        }
    ]
}

pdf_dir = lambda data_dir: f'{data_dir}/pdf'
generation_dir = lambda data_dir: f'{data_dir}/generation'
temp_dir = lambda data_dir: f'{data_dir}/temp'
evaluation_dir = lambda data_dir: f'{data_dir}/evaluation'

def load_doc_manager(doc_manager:DocManager, sample:Sample, load_from_pdf:bool, data_dir:str='.', simple_load:bool=False):
    if load_from_pdf:
        # Load from full pdf
        file_dir = pdf_dir(data_dir)
        source, doc_id = sample.doc_file.split(':', maxsplit=1)
        doc_file = f"{file_dir}/{doc_id}.pdf"
        outline_file = f"{file_dir}/outline_{doc_id}.txt"
        if not os.path.exists(doc_file):
            if source == ARXIV:
                download_arxiv_pdf(doc_id, doc_file)
            elif source == ACL:
                download_acl_pdf(doc_id, doc_file)
            else:
                raise ValueError(f'Unknown document source: {source}')
            
        if source == ARXIV:
            paragraphs, outline, title = get_arxiv_paper_text(doc_id)
            paragraphs = [DocManager.remove_citations(p) for p in paragraphs]
            doc_manager.load_doc(doc_strs=paragraphs, outline=outline, simple_load=simple_load)
        else:
            outline = None
            if os.path.exists(outline_file):
                with open(outline_file) as f:
                    outline = f.read()
            doc_manager.load_doc(doc_file=doc_file, outline=outline, simple_load=simple_load)
            if outline is None:
                with open(outline_file, 'w') as f:
                    f.write(doc_manager.full_outline)
    else:
        # Load from partial text
        doc_manager.load_doc(doc_strs=sample.doc_strs, outline=sample.outline, simple_load=simple_load)


def get_process_file(retrieval_method:str, prefix:str, sid:int, question_type:str, qid:int, load_from_pdf:bool, sent_chunk:bool, max_seq_len:int, k:int = None, is_temp:bool = False, data_dir:str=None):
    process_dir = temp_dir(data_dir) if is_temp else generation_dir(data_dir)
    if retrieval_method in {RAG, RAG_BASE}:
        return f'{process_dir}/{RAG}/{prefix}_{sid}_{question_type}_{qid}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json'
    elif retrieval_method in {GEN, CLS_GEN}:
        return f'{process_dir}/{retrieval_method}/{prefix}_{sid}_{question_type}_{qid}_{load_from_pdf}_{sent_chunk}_{max_seq_len}.json'
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')


def get_eval_file(retrieval_method:str, prefix:str, question_type:str, load_from_pdf:bool, sent_chunk:bool, max_seq_len:int, k:int = None, is_temp:bool = False, data_dir:str=None):
    eval_dir = temp_dir(data_dir) if is_temp else evaluation_dir(data_dir)
    if retrieval_method in {RAG, RAG_BASE}:
        return f"{eval_dir}/{retrieval_method}/{prefix}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json"
    elif retrieval_method in {GEN, CLS_GEN}:
        return f"{eval_dir}/{retrieval_method}/{prefix}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}.json"
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')


def parse_eval_file(eval_file:str, data_dir:str):
    if eval_file.startswith(temp_dir(data_dir)):
        eval_dir = temp_dir(data_dir)
        is_temp = True
    else:
        eval_dir = evaluation_dir(data_dir)
        is_temp = False
    retrieval_method, file_name = eval_file[len(eval_dir)+1:].split('/')
    file_name, _ = file_name.split('.')
    if retrieval_method in {RAG, RAG_BASE}:
        prefix, question_type, load_from_pdf, sent_chunk, max_seq_len, k = file_name.split('_')
        k:int = eval(k)
    elif retrieval_method in {GEN, CLS_GEN}:
        prefix, question_type, load_from_pdf, sent_chunk, max_seq_len = file_name.split('_')
        k = None
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    load_from_pdf:bool = eval(load_from_pdf)
    sent_chunk:bool = eval(sent_chunk)
    max_seq_len:int = eval(max_seq_len)
    return retrieval_method, prefix, question_type, load_from_pdf, sent_chunk, max_seq_len, k, is_temp


def get_sents_and_process(
    doc_manager:DocManager,
    retrieval_method:str,
    prefix:str,
    sid:int,
    question_type:str,
    load_from_pdf:bool,
    sent_chunk:bool,
    max_seq_len:int,
    k:int = None,
    is_temp:bool = False,
    data_dir:str=None
):
    process_file = get_process_file(retrieval_method, prefix, sid, question_type, load_from_pdf, sent_chunk, max_seq_len, k, is_temp, temp_dir(data_dir), generation_dir(data_dir))
    process = AgenticRAG.load_process(process_file)
    
    if retrieval_method == RAG:
        passages = list[str]()
        for step in process[::-1]:
            if AgenticRAG.TOOL_POST_PROCESS in step and step[AgenticRAG.TOOL_POST_PROCESS]:
                passages.extend(step[AgenticRAG.TOOL_POST_PROCESS][AgenticRAG.PASSAGES])
                break
        
    elif retrieval_method == GEN:
        chunk_ids = list(map(int, process[0][AgenticRAG.AGENT][AgenticRAG.MESSAGES][0].content.split(', ')))
        passages = [doc_manager.chunks[chunk_id].page_content for chunk_id in chunk_ids]
        
    elif retrieval_method == RAG_BASE:
        passages = list[str]()
        for step in process:
            if AgenticRAG.TOOL_POST_PROCESS in step and step[AgenticRAG.TOOL_POST_PROCESS]:
                passages.extend(passage.split(CHUNK_SEP)[1] for passage in step[AgenticRAG.TOOL_POST_PROCESS][AgenticRAG.RETRIEVAL])
        passages = list(set(passages))
    
    elif retrieval_method == CLS_GEN:
        lines:list[str] = process[0][AgenticRAG.AGENT][AgenticRAG.MESSAGES][0].content.split('\n')
        passages = list[str]()
        for chunk_id in range(len(doc_manager.chunks)):
            for line in lines:
                if line.startswith(f'{chunk_id}: ') or f' {chunk_id}: ' in line:
                    if 'yes' in line.lower():
                        passages.append(doc_manager.chunks[chunk_id].page_content)
                    break
    
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    
    retrieved_sents = passages if sent_chunk else [sent for passage in passages for sent in spacy_sent_tokenize(doc_manager.nlp, passage)]
    return process, retrieved_sents


def run_framework(
    agentic_rag:AgenticRAG,
    retrieval_method:str,
    prefix:str, 
    sid:int, 
    sample:Sample, 
    load_from_pdf:bool, 
    question_type:str, 
    sent_chunk:bool, 
    max_seq_len:int, 
    k:int,
    is_temp:bool = False,
    data_dir:str=None
    ):
    
    for qid, question in enumerate(sample.questions[question_type]):
        process_file = get_process_file(retrieval_method, prefix, sid, question_type, qid, load_from_pdf, sent_chunk, max_seq_len, k, is_temp, data_dir)
        if retrieval_method == RAG:
            agentic_rag.load_langgraph([RetrieveByDenseRetrieval(agentic_rag.doc_manager, k), RewriteQuestion])
            process = agentic_rag.invoke(question)
        elif retrieval_method == GEN:
            agentic_rag.load_langgraph()
            content = PARAGRAPH_SEP.join([f'Chunk {chunk.metadata["chunk_id"]}: {chunk.page_content}' for chunk in agentic_rag.doc_manager.chunks])
            prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nSelect the Chunk ids that are relevant to the following question: \n\n{question}\n\nReturn only the selected chunk ids separated by commas, e.g. "1, 3, 5".'
            process = agentic_rag.invoke(prompt)
        elif retrieval_method == CLS_GEN:
            agentic_rag.load_langgraph()
            content = PARAGRAPH_SEP.join([f'Chunk {chunk.metadata["chunk_id"]}: {chunk.page_content}' for chunk in agentic_rag.doc_manager.chunks])
            prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nFor each chunk, decide whether it is relevant to the following question: \n\n{question}\n\nReturn your decision in the following format:\n\n0: Yes\n1: No\n2: No\n...'
            process = agentic_rag.invoke(prompt)
        else:
            raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    
        AgenticRAG.dump_process(process, process_file)
