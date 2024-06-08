# from rank_bm25 import BM25Okapi
from nltk import word_tokenize
from nltk.corpus import wordnet

from .base import *

def read_jsonline(file:str):
    with open(file) as f_in:
        return [json.loads(l) for l in f_in]

def read_json(file:str):
    with open(file) as f_in:
        return json.load(f_in)
    
def write_json(file, obj):
    with open(file, 'w') as f_out:
        json.dump(obj, f_out)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class DocIndex:
    def __init__(self, graph:nx.DiGraph, paragraphs:List[str], summary:List[str], paragraph_embs:np.ndarray, pid2nodes:List[List[str]]) -> None:
        self.graph = graph
        self.paragraphs = paragraphs
        self.summary = summary
        self.paragraph_embs = paragraph_embs
        self.pid2nodes = pid2nodes
        # self.bm25 = BM25Okapi([[w.lower() for w in word_tokenize(p)] for p in paragraphs])
    

class ChunkInfo:
    def __init__(
        self, 
        cur_pid:int,
        passage:str, 
        summary:str='', 
        important_ents:List[str]=[], 
        ent_descriptions:Dict[str, str]={}, 
        relation_descriptions:List[Tuple[List[str], str]]=[], 
        prev_ent_descriptions:Dict[int, Dict[str, str]]={}, 
        prev_relation_descriptions:Dict[int, List[Tuple[List[str], str]]]={},
        prev_summaries:Dict[int, str]={}
        ) -> None:
        self.cur_pid = cur_pid
        self.passage = passage
        self.summary = summary
        self.important_ents = important_ents
        self.ent_descriptions = ent_descriptions
        self.relation_descriptions = relation_descriptions
        self.prev_ent_descriptions = {int(k): v for k, v in prev_ent_descriptions.items()}
        self.prev_relation_descriptions = {int(k): v for k, v in prev_relation_descriptions.items()}
        self.prev_summaries = {int(k): v for k, v in prev_summaries.items()}
        
    def to_json(self):
        return {
            'cur_pid': self.cur_pid,
            'passage': self.passage,
            'summary': self.summary,
            'important_ents': self.important_ents,
            'ent_descriptions': self.ent_descriptions,
            'relation_descriptions': self.relation_descriptions,
            'prev_ent_descriptions': self.prev_ent_descriptions,
            'prev_relation_descriptions': self.prev_relation_descriptions,
            'prev_summaries': self.prev_summaries
        }
    
    @property
    def recap_str(self):
        retrieved_recap = defaultdict(lambda: {'summary': '', 'ent_description': '', 'rel_description': ''})
        for pid, summary in self.prev_summaries.items():
            retrieved_recap[pid]['summary'] = summary
        for pid, e_d in self.prev_ent_descriptions.items():
            retrieved_recap[pid]['ent_description'] = '\n'.join([f'{e}: {d}' for e, d in e_d.items()])
        for pid, r_d in self.prev_relation_descriptions.items():
            retrieved_recap[pid]['rel_description'] = '\n'.join([f'{", ".join(r)}: {d}' for r, d in r_d])
        recaps = []
        retrieved_recap_list = list(retrieved_recap.items())
        retrieved_recap_list.sort(key=lambda x: x[0])
        for pid, retrieved in retrieved_recap_list:
            ent_d_str, rel_d_str, summary_str = '', '', ''
            if retrieved['summary']:
                summary_str = f"Summary:\n{retrieved['summary']}\n"
            if retrieved['ent_description']:
                ent_d_str = f"Entity descriptions:\n{retrieved['ent_description']}\n"
            if retrieved['rel_description']:
                rel_d_str = f"Relation descriptions:\n{retrieved['rel_description']}\n"
            recaps.append(f'Passage {int(pid) - self.cur_pid}:\n{ent_d_str}{rel_d_str}{summary_str}')
        recap_str = '\n'.join(recaps)
        return recap_str
        
    def print(self):
        important_ents_str = '\n'.join(self.important_ents)
        entity_description_str = '\n'.join([f'{e}: {d}' for e, d in self.ent_descriptions.items()])
        relation_description_str = '\n'.join([f'{r}: {d}' for r, d in self.relation_descriptions])
        print(f'''Recap:\n{self.recap_str}\n\nPassage:\n{self.passage}\n\nImportant entities:\n\n{important_ents_str}\n\nEntity descriptions:\n{entity_description_str}\n\nRelation description:\n{relation_description_str}\n\nSummary:\n{self.summary}''')
        

def log_info(log_file:str, tag:str, info):
    if log_file is not None:
        with open(log_file, 'a') as f_out:
            f_out.write(json.dumps([tag, info]))
            f_out.write('\n')
    

def get_synonym_pairs() -> List[Tuple[str, str]]:
    nouns = set()
    for noun in wordnet.all_synsets(wordnet.NOUN):
        nouns.update(noun.lemma_names())
    pairs = set()
    for noun in nouns:
        synonyms = {synset.name().split('.')[0].replace('_', ' ') : synset for synset in wordnet.synsets(noun, wordnet.NOUN)}
        if len(synonyms) < 2:
            continue
        pairs.update([frozenset((w1, w2)) for w1, w2 in itertools.combinations(synonyms.keys(), 2) if synonyms[w1].wup_similarity(synonyms[w2]) > 0.8])
    return [tuple(pair) for pair in pairs]
