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

class MyNode:
    def __init__(self, is_leaf:bool, level:int, index:int) -> None:
        self.is_leaf = is_leaf
        self.level = level
        self.index = index
        self.summary = ''
        self.children:List[MyNode | str] = []
        self.left_sibling:MyNode = None
        self.common_with_left = ''
        self.unique_to_left = ''
        self.unique_in_left = ''
        self.right_sibling:MyNode = None
        self.common_with_right = ''
        self.unique_to_right = ''
        self.unique_in_right = ''
        self.parent:MyNode = None
    
    def dump(self):
        return {
            'is_leaf': self.is_leaf,
            'level': self.level,
            'index': self.index,
            'summary': self.summary,
            'children': [child if isinstance(child, str) else child.get_location() for child in self.children],
            'left_sibling': self.left_sibling.get_location() if self.left_sibling is not None else None,
            'common_with_left': self.common_with_left,
            'unique_to_left': self.unique_to_left,
            'unique_in_left': self.unique_in_left,
            'right_sibling': self.right_sibling.get_location() if self.right_sibling is not None else None,
            'common_with_right': self.common_with_right,
            'unique_to_right': self.unique_to_right,
            'unique_in_right': self.unique_in_right,
            'parent': self.parent.get_location() if self.parent is not None else None
        }
        
    @classmethod
    def init_from_dict(cls, info:dict):
        obj:MyNode = cls(info['is_leaf'], info['level'], info['index'])
        obj.summary = info['summary']
        obj.children = info['children']
        obj.left_sibling = info['left_sibling']
        obj.common_with_left = info['common_with_left']
        obj.unique_to_left = info['unique_to_left']
        obj.unique_in_left = info['unique_in_left']
        obj.right_sibling = info['right_sibling']
        obj.common_with_right = info['common_with_right']
        obj.unique_to_right = info['unique_to_right']
        obj.unique_in_right = info['unique_in_right']
        obj.parent = info['parent']
        return obj
        
    def resolve_neighbors(self, tree:List[list]):
        if self.left_sibling is not None:
            self.left_sibling = tree[self.left_sibling[0]][self.left_sibling[1]]
        if self.right_sibling is not None:
            self.right_sibling = tree[self.right_sibling[0]][self.right_sibling[1]]
        if self.parent is not None:
            self.parent = tree[self.parent[0]][self.parent[1]]
        if self.children and not isinstance(self.children[0], str):
            self.children = [tree[level][index] for level, index in self.children]
        
    def get_location(self):
        return (self.level, self.index)

def dump_tree(tree_file:str, tree:List[List[MyNode]]):
    dumped_tree = [[node.dump() for node in level_nodes] for level_nodes in tree]
    write_json(tree_file, dumped_tree)
    
def load_tree(tree_file:str):
    dumped_tree = read_json(tree_file)
    tree = [[MyNode.init_from_dict(info) for info in level_infos] for level_infos in dumped_tree]
    for level_nodes in tree:
        for node in level_nodes:
            node.resolve_neighbors(tree)
    return tree