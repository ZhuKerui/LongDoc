from sci_review.data_base import *
import jsonlines
from sci_review.agentic_rag import *
from tqdm import tqdm


ACLSUM_DIR = '../../data/ACLSum'
ACLSUM_PDF_DIR = f'{ACLSUM_DIR}/pdf'
ACLSUM_GENERATION_DIR = f'{ACLSUM_DIR}/generation'
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
    k:int
    ):
    
    if retrieval_method == 'rag':
        agentic_rag.load_langgraph([RetrieveByDenseRetrieval(agentic_rag.doc_manager, k), RewriteQuestion])
        process = agentic_rag.invoke(sample.questions[question_type])
        process_file = f'{ACLSUM_GENERATION_DIR}/{retrieval_method}/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json'
        AgenticRAG.dump_process(process, process_file)
    elif retrieval_method == 'gen':
        agentic_rag.load_langgraph()
        content = PARAGRAPH_SEP.join([f'Chunk {chunk.metadata["chunk_id"]}: {chunk.page_content}' for chunk in agentic_rag.doc_manager.chunks])
        prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nSelect the Chunk ids that are relevant to the following question: \n\n{sample.questions[question_type]}\n\nReturn only the selected chunk ids separated by commas, e.g. "1, 3, 5".'
        process = agentic_rag.invoke(prompt)
        process_file = f'{ACLSUM_GENERATION_DIR}/{retrieval_method}/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}.json'
        AgenticRAG.dump_process(process, process_file)
        
if __name__ == '__main__':
    
    # Dataset config
    split = 'train'
    load_from_pdf = False
    # load_from_pdf = True

    # Retrieval config
    retrieval_method = 'rag'
    # retrieval_method = 'gen'

    # Chunk config
    # sent_chunk = True
    # max_seq_len = None
    # k = 10
    # sent_chunk = False
    # max_seq_len = None
    # k = 3
    sent_chunk = False
    max_seq_len = 100
    k = 10
    
    
    with open('words_alpha.txt') as f:
        words_alpha = set(f.read().splitlines())
    doc_manager = DocManager(word_vocab=words_alpha)

    agentic_rag = AgenticRAG()
    agentic_rag.doc_manager = doc_manager


    with jsonlines.open(f'{ACLSUM_DIR}/{split}_dataset.jsonl') as f_in:
        aclsum_dataset = [Sample.model_validate(line) for line in f_in]
        for sid in tqdm(range(0, 100)):
            print(sid)
            if sid in [60, 70, 76]:
                continue
            sample = aclsum_dataset[sid]
            if sample.doc_file.split('/')[-1] in bad_pdfs:
                continue
            
            load_doc_manager(agentic_rag.doc_manager, sample, load_from_pdf)
            agentic_rag.doc_manager.build_chunks(sent_chunk=sent_chunk, max_seq_length=max_seq_len)
            
            for question_type in ['challenge', 'approach', 'outcome']:
                run_aclsum(
                    agentic_rag=agentic_rag, 
                    retrieval_method=retrieval_method, 
                    split=split, 
                    sid=sid, 
                    sample=sample, 
                    load_from_pdf=load_from_pdf, 
                    question_type=question_type, 
                    sent_chunk=sent_chunk, 
                    max_seq_len=max_seq_len, 
                    k=k, 
                )
