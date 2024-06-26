from tqdm.notebook import tqdm
from nltk import sent_tokenize
from transformers import AutoTokenizer
import sys
import seaborn as sb
sys.path.append('../..')

from src import *
from src.test_utils import *

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# gritlm = GritLM("GritLM/GritLM-7B", device_map="cuda:2", torch_dtype="auto")
retriever = Retriever(device='cpu', syn_dist=0.1)
doc_split = DocSplit(retriever.retriever_tokenizer)
# llm = LLM()
llm = 'mistralai/Mistral-7B-Instruct-v0.2'
# llm = None
longdoc = LongDoc(retriever, llm)
# dataset = NarrativeQADataset(llm)
dataset = QualityDataset(llm, split='dev')
# reading_agent = ReadingAgent(dataset, llm)

test_i = 2
sample = dataset.data[test_i]
questions, answers = dataset.get_questions_and_answers(sample)
article = dataset.get_article(sample)
questions = [q.splitlines()[0] for q in questions]
questions

pages = doc_split.split_paragraphs(article, 50)
all_summary = read_json('all_summary.json')

tree = longdoc.build_summary_pyramid(pages, all_summary)
dump_tree('temp_tree.json', tree)