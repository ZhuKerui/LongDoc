import sys
sys.path.append('../..')

from datetime import datetime
from langchain_core.tracers.context import tracing_v2_enabled

import os
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


from src import *
from src.test_utils import *
from src.summary_tree import *
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

dataset = QualityDataset(None, split='dev')
# dataset = NarrativeQADataset()

f = Factory()

for test_id in tqdm(range(1, 20)):
    ret_tool = 'dpr'
    article = dataset.get_article(dataset.data[test_id])
    questions, answers = dataset.get_questions_and_answers(dataset.data[test_id])
    dpr_retriever, tree_retriever, documents = f.build_corpus(article, dpr_file=os.path.join(dataset.data_dir, f'dpr_{test_id}.json'), tree_file=os.path.join(dataset.data_dir, f'tree_{test_id}.json'))

    agent = NavigateAgent(f.llm)
    app = agent.create_workflow(dpr_retriever=dpr_retriever, tree_retriever=tree_retriever, documents=documents)
    date = datetime.fromtimestamp(time())
    project_name = f'NavigateAgent-{ret_tool}-{dataset.dataset_name}-{test_id}-{date.month}/{date.day}-{date.hour}:{date.minute}'
    with tracing_v2_enabled(project_name=project_name):
        for qid, q in enumerate(questions):
            try:
                app.invoke(NavigateState(question=q, retriever=ret_tool, answer_file=os.path.join(dataset.data_dir, f'gen_{test_id}_{qid}.json')))
            except:
                continue
