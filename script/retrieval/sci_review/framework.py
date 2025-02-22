
from .agentic_rag import AgenticRAG
from .chunk_selection import ChunkSelection
from .citation_generation import CitationGeneration
from .data_base import *
from .pipeline_base import *


class RetrievalMethod(Enum):
    RAG = 'rag'
    GEN = 'gen'
    RAG_BASE = 'rag_base'
    CLS = 'cls'
    CITATION = 'citation'
    
retrieval_method_pipeline:dict[RetrievalMethod, Type[MyPipeline]] = {
    RetrievalMethod.RAG: AgenticRAG,
    RetrievalMethod.GEN: ChunkSelection,
    RetrievalMethod.RAG_BASE: AgenticRAG,
    RetrievalMethod.CLS: ChunkSelection,
    RetrievalMethod.CITATION: CitationGeneration
}

pdf_dir = lambda data_dir: f'{data_dir}/pdf'
generation_dir = lambda data_dir: f'{data_dir}/generation'
temp_dir = lambda data_dir: f'{data_dir}/temp'
evaluation_dir = lambda data_dir: f'{data_dir}/evaluation'

def get_process_file(retrieval_method:RetrievalMethod, prefix:str, sid:int, question_type:str, qid:int, k:int = None, is_temp:bool = False, data_dir:str=None):
    process_dir = temp_dir(data_dir) if is_temp else generation_dir(data_dir)
    if retrieval_method in {RetrievalMethod.RAG, RetrievalMethod.RAG_BASE}:
        return f'{process_dir}/{RetrievalMethod.RAG.value}/{prefix}_{sid}_{question_type}_{qid}_{k}.json'
    else:
        return f'{process_dir}/{retrieval_method.value}/{prefix}_{sid}_{question_type}_{qid}.json'


def get_eval_file(retrieval_method:RetrievalMethod, prefix:str, question_type:str, k:int = None, is_temp:bool = False, data_dir:str=None):
    eval_dir = temp_dir(data_dir) if is_temp else evaluation_dir(data_dir)
    if retrieval_method in {RetrievalMethod.RAG, RetrievalMethod.RAG_BASE}:
        return f"{eval_dir}/{retrieval_method.value}/{prefix}_{question_type}_{k}.json"
    else:
        return f"{eval_dir}/{retrieval_method.value}/{prefix}_{question_type}.json"


def parse_eval_file(eval_file:str, data_dir:str):
    if eval_file.startswith(temp_dir(data_dir)):
        eval_dir = temp_dir(data_dir)
        is_temp = True
    else:
        eval_dir = evaluation_dir(data_dir)
        is_temp = False
    retrieval_method, file_name = eval_file[len(eval_dir)+1:].split('/')
    file_name, _ = file_name.split('.')
    if retrieval_method in {RetrievalMethod.RAG, RetrievalMethod.RAG_BASE}:
        prefix, question_type, k = file_name.split('_')
        k:int = eval(k)
    elif retrieval_method in {RetrievalMethod.GEN, RetrievalMethod.CLS}:
        prefix, question_type = file_name.split('_')
        k = None
    else:
        raise ValueError(f'Unknown retrieval method: {retrieval_method}')
    return retrieval_method, prefix, question_type, k, is_temp


def run_framework(
    doc_manager:DocManager,
    retrieval_method:RetrievalMethod,
    prefix:str, 
    sid:int, 
    sample:Sample, 
    question_type:str, 
    is_temp:bool,
    data_dir:str,
    k:int,
    llm_model:str = GPT_MODEL_CHEAP,
    tool_list:list[BaseTool] = None
    ):
    
    for qid, question in enumerate(sample.questions[question_type]):
        process_file = get_process_file(
            retrieval_method=retrieval_method,
            prefix=prefix,
            sid=sid,
            question_type=question_type,
            qid=qid,
            k=k,
            is_temp=is_temp,
            data_dir=data_dir
        )
        agent_pipeline_cls = retrieval_method_pipeline[retrieval_method]
        agent_pipeline = agent_pipeline_cls(llm_model=llm_model)
        match retrieval_method:
            case RetrievalMethod.RAG:
                assert tool_list, 'Tool list must be provided for RAG retrieval method.'
                agent_pipeline.load_langgraph(tools=tool_list)
            case RetrievalMethod.GEN:
                agent_pipeline.load_langgraph(ChunkSelection.ChunkSelectionType.SELECTION)
            case RetrievalMethod.CLS:
                agent_pipeline.load_langgraph(ChunkSelection.ChunkSelectionType.CLASSIFICATION)
            case RetrievalMethod.CITATION:
                agent_pipeline.load_langgraph()
        
        process = agent_pipeline.invoke(question, [chunk.page_content for chunk in doc_manager.chunks])
        sample.relevant_blocks[question] = agent_pipeline_cls.get_chunk_ids_from_process(process)
        agent_pipeline_cls.dump_process(process, process_file)
        if retrieval_method == RetrievalMethod.CITATION:
            sample.generated_extractions[question] = agent_pipeline_cls.get_extraction_from_process(process)
    
