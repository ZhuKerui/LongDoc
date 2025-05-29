
from .agentic_rag import AgenticRAG
from .chunk_selection import ChunkSelection
from .citation_generation import CitationGeneration
from .gt_collection import GTCollection
from .data_base import *
from .pipeline_base import *


class RetrievalMethod(Enum):
    RAG = 'rag'
    GEN = 'gen'
    RAG_BASE = 'rag_base'
    CLS = 'cls'
    CITATION = 'citation'
    EXPLANATION = 'explanation'
    
retrieval_method_pipeline:dict[RetrievalMethod, Type[MyPipeline]] = {
    RetrievalMethod.RAG: AgenticRAG,
    RetrievalMethod.GEN: ChunkSelection,
    RetrievalMethod.RAG_BASE: AgenticRAG,
    RetrievalMethod.CLS: ChunkSelection,
    RetrievalMethod.CITATION: CitationGeneration,
    RetrievalMethod.EXPLANATION: GTCollection
}

pdf_dir = lambda data_dir: f'{data_dir}/pdf'
generation_dir = lambda data_dir: f'{data_dir}/generation'
temp_dir = lambda data_dir: f'{data_dir}/temp'
evaluation_dir = lambda data_dir: f'{data_dir}/evaluation'

def get_process_file(retrieval_method:RetrievalMethod, prefix:str, sid:int, question_type:str, qid:int, k:int = None, is_temp:bool = False, data_dir:str=None):
    process_dir = temp_dir(data_dir) if is_temp else generation_dir(data_dir)
    retrieval_setting = f'{retrieval_method.value}_{k}'
    return f'{process_dir}/{retrieval_setting}/{prefix}_{sid}_{question_type}_{qid}.json'


# def get_eval_file(retrieval_method:RetrievalMethod, prefix:str, question_type:str, k:int = None, is_temp:bool = False, data_dir:str=None):
#     eval_dir = temp_dir(data_dir) if is_temp else evaluation_dir(data_dir)
#     if retrieval_method in {RetrievalMethod.RAG, RetrievalMethod.RAG_BASE}:
#         return f"{eval_dir}/{retrieval_method.value}/{prefix}_{question_type}_{k}.json"
#     else:
#         return f"{eval_dir}/{retrieval_method.value}/{prefix}_{question_type}.json"


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
            case RetrievalMethod.EXPLANATION:
                agent_pipeline.load_langgraph()
        
        if retrieval_method == RetrievalMethod.EXPLANATION:
            process = agent_pipeline.invoke(question=question, answer=sample.answers[question], chunks=[chunk.page_content for chunk in doc_manager.chunks])
        else:
            process = agent_pipeline.invoke(question, [chunk.page_content for chunk in doc_manager.chunks])
        sample.selected_blocks[question] = agent_pipeline_cls.get_chunk_ids_from_process(process)
        agent_pipeline_cls.dump_process(process, process_file)
        if retrieval_method in {RetrievalMethod.CITATION, RetrievalMethod.EXPLANATION}:
            sample.generated_extractions[question] = agent_pipeline_cls.get_extraction_from_process(process)
    
def eval_selected_blocks(dataset:list[Sample], eval_metrics:EvalMetrics) -> list:
    for sample in dataset:
        for _, questions in sample.questions.items():
            for question in questions:
                relevant_blocks_gold = [int(i in sample.relevant_blocks[question] and sample.doc_blocks_with_label[i]) for i in range(len(sample.doc_blocks))]
                selected_blocks_pred = [int(i in sample.selected_blocks[question] and sample.doc_blocks_with_label[i]) for i in range(len(sample.doc_blocks))]
                sample.selected_blocks_eval[question] = eval_metrics.eval_precision_recall_f1(predictions=selected_blocks_pred, references=relevant_blocks_gold) if sum(selected_blocks_pred) > 0 else {'precision':0, 'recall':0, 'f1':0}

def print_eval_selected_blocks(dataset:list[Sample], eval_by_qtype:bool = False):
    qtype2metric2scores:dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list[float]))
    for sample in dataset:
        for qtype, questions in sample.questions.items():
            for question in questions:
                for metric, score in sample.selected_blocks_eval[question].items():
                    qtype2metric2scores[qtype if eval_by_qtype else ''][metric].append(score)
        
    for qtype, metric2scores in qtype2metric2scores.items():
        print(qtype)
        for metric, scores in metric2scores.items():
            print(metric, np.mean(scores))
        print()
    print()