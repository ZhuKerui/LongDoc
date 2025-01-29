from sci_review.data_base import *
import jsonlines
from sci_review.agentic_rag import *
from tqdm import tqdm
import pickle


# Dataset config
split = 'train'
load_from_pdf = False
question_type = 'challenge'

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


with jsonlines.open(f'../../data/ACLSum/{split}_dataset.jsonl') as f_in:
    aclsum_dataset = [Sample.model_validate(line) for line in f_in]
    for sid, sample in enumerate(tqdm(aclsum_dataset)):
        if load_from_pdf:
            # Load from full pdf
            doc_file = f"../../data/ACLSum/{sample.doc_file.split('/')[-1]}"
            outline_file = f"../../data/ACLSum/generation/outline_{sample.doc_file.split('/')[-1].replace('.pdf', '.txt')}"
            if not os.path.exists(doc_file):
                download_file(sample.doc_file, doc_file)
            if os.path.exists(outline_file):
                with open(outline_file) as f:
                    outline = f.read()
            else:
                outline = None
            doc_manager.load_doc(doc_file, outline)
            if not outline:
                with open(outline_file, 'w') as f:
                    f.write(doc_manager.full_outline)
        else:
            # Load from partial text
            doc_manager.load_doc(doc_strs=sample.doc_strs, outline=sample.outline)
            
        unique_ngram2sent = get_sent_index([sent.text for sent in doc_manager.sents])
        doc_manager.build_chunks(sent_chunk=sent_chunk, max_seq_length=max_seq_len)
        agentic_rag.load_langgraph([RetrieveByDenseRetrieval(doc_manager, k), RewriteQuestion])
        process = agentic_rag.invoke(sample.questions[question_type])
        try:
            process_file = f'../../data/ACLSum/generation/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.json'
            AgenticRAG.dump_process(process, process_file)
        except:
            process_file = f'../../data/ACLSum/generation/{split}_{sid}_{question_type}_{load_from_pdf}_{sent_chunk}_{max_seq_len}_{k}.pickle'
            with open(process_file, 'wb') as f:
                pickle.dump(process, f)