# !wget https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
import time, datetime, json, os
from tqdm import tqdm
from collections import defaultdict

from index_files import LongDoc, write_json, QualityDataset, NarrativeQADataset, LooGlEDataset, MuSiQueDataset, ReadingAgent, read_json, read_jsonline, LLMServer

llm_server = LLMServer()

# dataset = NarrativeQADataset()
# dataset = LooGlEDataset()
# dataset = QualityDataset(split='dev')
dataset = MuSiQueDataset()
reading_agent = ReadingAgent(dataset)
longdoc = LongDoc(dataset, device='cpu')

start = 0
end = 20

import sys

for eid in tqdm(range(start, end)):
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
                write_json(os.path.join(dataset.data_dir, f'rel_index_{eid}.json'), longdoc.index_text(['\n'.join(page) for page in pages]))