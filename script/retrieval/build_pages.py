# !wget https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
import sys
sys.path.append('../..')

from src.base import *
from src.base_utils import write_json, read_json
from src.data import QualityDataset, NarrativeQADataset, LooGlEDataset, MuSiQueDataset
from src.models import LLMServer, Retriever
from src.index_files import ReadingAgent, LongDoc

llm_server = LLMServer()
retriever = Retriever()

# dataset = NarrativeQADataset()
# dataset = LooGlEDataset()
dataset = QualityDataset(split='dev')
# dataset = MuSiQueDataset()
reading_agent = ReadingAgent(dataset)
# longdoc = LongDoc(dataset, retriever)
longdoc = LongDoc(retriever)

start = 0
end = 40
w_note = True
r_num = 2

import sys

for eid in range(start, end):
    print(eid - start+ 1, '/', end - start)
    page_file = os.path.join(dataset.data_dir, f'pages_{eid}.json')
    if sys.argv[1] == 'page':
        if not os.path.exists(page_file):
            write_json(page_file, dataset.pagination(dataset.get_article(dataset.data[eid]), verbose=False))
        
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
                    file_name = f'index_w_{r_num}_{eid}.json'
                else:
                    file_name = f'index_{eid}.json'
                write_json(os.path.join(dataset.data_dir, file_name), [ci.to_json() for ci in longdoc.index_text(['\n'.join(page) for page in pages], w_note, r_num)])