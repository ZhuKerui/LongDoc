# !wget https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
import sys
sys.path.append('../..')

from src.base import *
from src.base_utils import write_json, read_json
from src.data import QualityDataset, NarrativeQADataset, LooGlEDataset, MuSiQueDataset
from src.summary_tree import Factory

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


# dataset = NarrativeQADataset()
# dataset = LooGlEDataset()
dataset = QualityDataset(llm=None, split='dev')
factory = Factory()
# dataset = MuSiQueDataset()

# if sys.argv[1] == 'gist':
#     reading_agent = ReadingAgent(dataset, llm)
# longdoc = LongDoc(retriever, llm)

start = 0
end = 20

import sys

for eid in range(start, end):
    print(eid - start+ 1, '/', end - start)
    dpr_file = os.path.join(dataset.data_dir, f'dpr_{eid}.json')
    tree_file = os.path.join(dataset.data_dir, f'tree_{eid}.json')
    article = dataset.get_article(dataset.data[eid])
    _, _ = factory.build_corpus(text=article, dpr_file=dpr_file, tree_file=tree_file)
    