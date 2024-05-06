# !wget https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
import sys
sys.path.append('../..')

from src.base import *
from src.base_utils import write_json, read_json
from src.data import QualityDataset, NarrativeQADataset, LooGlEDataset, MuSiQueDataset
from src.models import LLMServer, Retriever, LLM
from src.index_files import ReadingAgent, LongDoc, DocSplit

# llm = LLM(device_map='cuda:1')
llm = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = None
retriever = Retriever(device='cuda:2')

# dataset = NarrativeQADataset()
# dataset = LooGlEDataset()
dataset = QualityDataset(llm, split='dev')
# dataset = MuSiQueDataset()

if sys.argv[1] == 'gist':
    reading_agent = ReadingAgent(dataset, llm)
longdoc = LongDoc(retriever, llm)
doc_split = DocSplit('intfloat/multilingual-e5-large')

start = 3
end = 10
w_note = True
r_num = 1
match_num = 2

import sys

for eid in range(start, end):
    print(eid - start+ 1, '/', end - start)
    page_file = os.path.join(dataset.data_dir, f'pages_{eid}.json')
    if sys.argv[1] == 'page':
        if not os.path.exists(page_file):
            # write_json(page_file, dataset.pagination(dataset.get_article(dataset.data[eid]), verbose=False))
            write_json(page_file, doc_split.split_paragraphs(dataset.get_article(dataset.data[eid]), 500))
        
    else:
        if os.path.exists(page_file):
            pages = read_json(page_file)
            if sys.argv[1] == 'gist':
                gist_file = os.path.join(dataset.data_dir, f'gist_{eid}.json')
                if not os.path.exists(gist_file):
                    example_with_gists = reading_agent.gisting(dataset.data[eid], pages, verbose=False)
                    gist = {'word_count': example_with_gists['word_count'], 'gist_word_count': example_with_gists['gist_word_count'], 'shortened_pages': example_with_gists['shortened_pages']}
                    write_json(gist_file, gist)
        
            elif sys.argv[1] == 'index':
                if w_note:
                    file_name = f'index_wg_{match_num}_{r_num}_{eid}.json'
                else:
                    file_name = f'index_{eid}.json'
                write_json(os.path.join(dataset.data_dir, file_name), [ci.to_json() for ci in longdoc.index_text(pages, w_note, match_num, r_num)])