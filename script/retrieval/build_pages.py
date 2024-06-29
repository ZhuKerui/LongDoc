# !wget https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
import sys
sys.path.append('../..')

from src.data import QualityDataset, NarrativeQADataset, LooGlEDataset, MuSiQueDataset
from src.index_files import *

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


# dataset = NarrativeQADataset()
# dataset = LooGlEDataset()
dataset = QualityDataset(split='dev')
# dataset = MuSiQueDataset()

start = 0
end = 40

factory = Factory(device='cuda:2', chunk_size=100)
longdoc = LongDoc(factory=factory)

for eid in range(start, end):
    print(eid - start+ 1, '/', end - start)
    index_file = os.path.join(dataset.data_dir, f'index_{eid}.json')
    article = dataset.get_article(dataset.data[eid])
    longdoc.build_index(article, chunk_info_file=index_file)
    