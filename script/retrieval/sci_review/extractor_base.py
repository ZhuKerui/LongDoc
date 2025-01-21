# from tools import *
# from tools.TextProcessing import remove_brackets
# from tools.transformer_utils import *
# from tools.db_helper import *
import torch
from statistics import mean

database_host = os.environ['COVIDDEER_DATABASE_HOST']

class IdGenerator:
    def __init__(self):
        self.d: Dict[Any, int] = {}
        self.l: List[Any] = []
    
    
    def get_id(self, obj, update:bool=True):
        ret = self.d.get(obj)
        if ret is not None:
            return ret
        elif update:
            ret = len(self.d)
            self.d[obj] = ret
            self.l.append(obj)
            return ret
        else:
            return None
    
    
    def get_obj(self, id:int):
        return self.l[id]
        
        
    def save(self, dump_file:str):
        write_json(dump_file, self.l)
    
    
    def load(self, load_file:str):
        self.l = [item if type(item) != list else tuple(item) for item in read_json(load_file)]
        self.d = {item : i for i, item in enumerate(self.l)}
        
        
    def load_list(self, load_list:List[Any]):
        self.l = [item if type(item) != list else tuple(item) for item in load_list]
        self.d = {item : i for i, item in enumerate(self.l)}



class SentenceAnalyzerBase:
    def __init__(self, spacy_nlp:Language=None, corepath_file:str=None, subpath_file:str=None, corepath_cnt_file:str=None, subpath_cnt_file:str=None):
        self.subpath_id_generator = IdGenerator()
        self.corepath_id_generator = IdGenerator()
        self.corepath_counter = Counter()
        self.subpath_counter = Counter()
        self.spacy_nlp = spacy_nlp
        
        if corepath_file is not None and os.path.exists(corepath_file):
            self.corepath_id_generator.load(corepath_file)
        if subpath_file is not None and os.path.exists(subpath_file):
            self.subpath_id_generator.load(subpath_file)
        if corepath_cnt_file is not None and os.path.exists(corepath_cnt_file):
            for k, v in read_csv(corepath_cnt_file):
                self.corepath_counter[int(k)] = int(v)
            self.update_explicitness(update_core=True, update_sub=False)
            
        if subpath_cnt_file is not None and os.path.exists(subpath_cnt_file):
            for k, v in read_csv(subpath_cnt_file):
                self.subpath_counter[int(k)] = int(v)
            self.update_explicitness(update_core=False, update_sub=True)
            
    # --------------------------------------------------------- Static functions start ---------------------------------------------------------
    @staticmethod
    def get_back(token:Token):
        '''
        Get the index of the last token of an entity name in the SpaCy document
        '''
        while token.dep_ == 'compound':
            token = token.head
        return token

    @staticmethod
    def get_front(token:Token):
        '''
        Get the index of the first token of an entity name in the SpaCy document
        '''
        mod_exist = True
        temp_token = token
        while mod_exist:
            token = temp_token
            mod_exist = False
            for c in token.children:
                if 'compound' == c.dep_:
                    if c.i < temp_token.i:
                        temp_token = c
                        mod_exist = True
        return token

    @staticmethod
    def get_path(doc:Doc, kw1_steps:List[int], kw2_steps:List[int]):
        '''
        Collect the corepath in str
        '''
        path_tokens = []
        for step in kw1_steps:
            path_tokens.append('i_' + doc[step].dep_)
        kw2_steps.reverse()
        for step in kw2_steps:
            path_tokens.append(doc[step].dep_)
        return ' '.join(path_tokens)

    @staticmethod
    def reverse_path(path:str):
        path = path.split()
        r_path = ' '.join(['i_' + token if token[:2] != 'i_' else token[2:] for token in reversed(path)])
        return r_path
    
    @staticmethod
    def collect_sub_dependency_path(doc:Doc, branch:np.ndarray):
        paths:List[Tuple[int, str, int]] = []
        dep_path:List[int] = np.arange(*branch.shape)[branch!=0]
        for token_id in dep_path:
            temp_paths = [(token_id, child.dep_, child.i) for child in doc[token_id].children if branch[child.i] == 0]
            while len(temp_paths) > 0:
                item  = temp_paths.pop()
                paths.append(item)
                temp_paths.extend([(item[0], item[1] + ' ' + child.dep_, child.i) for child in doc[item[2]].children if branch[child.i] == 0])
        return paths
    
    @staticmethod
    def gen_subpath_pattern(path:str):
        return ' '.join(path.replace('compound', '').replace('conj', '').replace('appos', '').split())
    
    @staticmethod
    def tokenize_ent(ents:List[str]):
        clean_ent = [remove_brackets(e).split(',')[0].strip() for e in ents]
        return [text if text else ents[i] for i, text in enumerate(clean_ent)]
    
    @staticmethod
    def normalize_noun_phrase(doc:Doc):
        return ' '.join([(token.lemma_ if token.pos_ == 'NOUN' else token.text).lower() for token in doc])
    
    @staticmethod
    def gen_corepath_pattern(path:str):
        if 'i_nsubj' not in path:
            path = SentenceAnalyzerBase.reverse_path(path)
        path = path.split()
        path_ = []
        # Check for 'prep prep'
        for token_idx, token in enumerate(path):
            if 'appos' in token or 'conj' in token:
                continue
            if token_idx > 0:
                if token == 'prep' and path[token_idx - 1] == 'prep':
                    continue
            path_.append(token)
        return ' '.join(path_)
    
    @staticmethod
    def get_phrase_full_span(phrase_span:Span):
        '''
        Get the complete span of a phrase in the SpaCy document
        '''
        phrase_right_most_token = SentenceAnalyzerBase.get_back(phrase_span[-1])
        phrase_left_most_token = min([SentenceAnalyzerBase.get_front(phrase_right_most_token), SentenceAnalyzerBase.get_front(phrase_span[0])], key=lambda x:x.i)
        return phrase_left_most_token, phrase_right_most_token
    
    @staticmethod
    def find_dependency_info_from_tree(doc:Union[Doc, Span], kw1:Span, kw2:Span):
        '''
        Find the dependency path that connect two entity names in the SpaCy document

        ## Return
            The steps from entity one to entity two or to the sub-root node
            The steps from entity two to entity two or to the sub-root node
            The corepath connecting the two entities
        '''
        # Find roots of the spans
        offset = doc[0].i
        kw1_front, kw1_end = kw1[0].i, kw1[-1].i
        kw2_front, kw2_end = kw2[0].i, kw2[-1].i
        branch = np.zeros(len(doc))
        kw1_steps = []
        kw2_steps = []
        path_found = False
        
        # Start from entity one
        i = kw1.root.i
        while branch[i-offset] == 0:
            branch[i-offset] = 1
            kw1_steps.append(i-offset)
            i = doc[i-offset].head.i
            if i >= kw2_front and i <= kw2_end:
                # entity two is the parent of entity one
                path_found = True
                break
            
        if not path_found:
            # If entity two is not the parent of entity one, we start from entity two
            i = kw2.root.i
            while branch[i-offset] != 1:
                branch[i-offset] = 2
                kw2_steps.append(i-offset)
                if i == doc[i-offset].head.i:
                    # If we reach the root of the tree, which hasn't been visited by the path from entity one, 
                    # it means entity one and two are not in the same tree, no path is found
                    return [], [], np.zeros(len(doc))
                
                i = doc[i-offset].head.i
                if i >= kw1_front and i <= kw1_end:
                    # entity one is the parent of entity two
                    branch[branch != 2] = 0
                    kw1_steps = []
                    path_found = True
                    break
        
        if not path_found:
            # entity one and entity two are on two sides, i is their joint
            break_point = kw1_steps.index(i-offset)
            branch[kw1_steps[break_point+1 : ]] = 0
            kw1_steps = kw1_steps[:break_point] # Note that we remain the joint node in the branch, but we don't include joint point in kw1_steps and kw2_steps
                                                # this is because the joint node is part of the path and we need the modification information from it, 
                                                # but we don't care about its dependency
        branch[branch != 0] = 1             # Unify the branch to contain only 0s and 1s
        branch[kw1_front-offset : kw1_end+1-offset] = 1   # Mark the entity one as part of the branch
        branch[kw2_front-offset : kw2_end+1-offset] = 1   # Mark the entity two as part of the branch
        return kw1_steps, kw2_steps, branch
    
    @staticmethod
    def sentence_decompose(doc:Doc, kw1:Span, kw2:Span, check_full_span:bool=True):
        '''
        Analyze the sentence with two entity names

        ## Return
        List of tuples. The returned tuples satisfy that:
            1. There exists a corepath starting with 'i_nsubj"
            2. The entity names are the complete span itself
        Each tuple contains the following fields: 
            span of entity one
            span of entity two
            a numpy array indicate the corepath
            the corepath in str
            the pattern generated from corepath
        '''

        if check_full_span:
            kw1_left_most_token, kw1_right_most_token = SentenceAnalyzerBase.get_phrase_full_span(kw1)
            kw2_left_most_token, kw2_right_most_token = SentenceAnalyzerBase.get_phrase_full_span(kw2)
            if kw1_left_most_token.i != kw1[0].i or kw1_right_most_token.i != kw1[-1].i or kw2_left_most_token.i != kw2[0].i or kw2_right_most_token.i != kw2[-1].i:
                # full span and keyword span don't match
                return None
        kw1_steps, kw2_steps, branch = SentenceAnalyzerBase.find_dependency_info_from_tree(doc, kw1, kw2)
        if not branch.any():
            # If the branch is empty, it means no corepath is found
            return None
        path = SentenceAnalyzerBase.get_path(doc, kw1_steps, kw2_steps)
        if path.count('nsubj') != 1:
            return None
        subj, obj = (kw1, kw2) if 'i_nsubj' in path else (kw2, kw1)
        pattern = SentenceAnalyzerBase.gen_corepath_pattern(path)
        if not pattern.startswith('i_nsubj'): # or not pattern.endswith('obj'):
            # If the corepath does not start with 'i_nsubj', we drop it
            return None
        
        return subj, obj, branch, pattern
    
    @staticmethod
    def collect_pattern(doc:Doc, kw1:Union[Span, Tuple[int, int]], kw2:Union[Span, Tuple[int, int]], check_full_span:bool=True):
        '''
        Collect subpath pattern between two entities from a SpaCy document

        ## Return
            List of corepath and subpath patterns collected from this document between the two entities
        '''
        if isinstance(kw1, tuple):
            kw1 = doc[kw1[0]:kw1[1]]
        if isinstance(kw2, tuple):
            kw2 = doc[kw2[0]:kw2[1]]
            
        ret = SentenceAnalyzerBase.sentence_decompose(doc, kw1, kw2, check_full_span)
        if ret is None:
            return None
        subj, obj, branch, pattern = ret
        sub_patterns:List[str] = []
        subPaths = defaultdict(set)
        for item in SentenceAnalyzerBase.collect_sub_dependency_path(doc, branch):
            if 'punct' in item[1]:
                continue
            sub_pattern = SentenceAnalyzerBase.gen_subpath_pattern(item[1])
            if not sub_pattern:
                continue
            subPaths[item[0]].add(sub_pattern)
        for v in subPaths.values():
            sub_patterns.extend(list(v))

        return subj, obj, pattern, sub_patterns
        
    @staticmethod
    def batch_collect_pattern(doc:Doc, pairs:Union[Iterable[Tuple[Span, Span]], Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]], check_full_span:bool=True):
        info:List[Tuple[Span, Span, str, List[str]]] = []
        for span1, span2 in pairs:
            patterns = SentenceAnalyzerBase.collect_pattern(doc, span1, span2, check_full_span)
            if patterns is not None:
                info.append(patterns)
        return info
    
    @staticmethod
    def is_subj(span:Span):
        root_token = span.root
        while root_token.dep_ in ['appos', 'conj']:
            root_token = root_token.head
        return root_token.dep_.startswith('nsubj')

    @staticmethod
    def find_span(doc:Doc, phrase:Doc, use_lemma:bool=False, lower:bool=False):
        """
        Find all the occurances of a given phrase in the sentence using spacy.tokens.doc.Doc
        Inputs
        ----------
        doc : spacy.tokens.doc.Doc
            a doc analyzed by a spacy model
        phrase : spacy.tokens.doc.Doc
            the phrase to be searched
        use_lemma : bool
            if true, the lemmazation is applied to each token before searching
        Return
        -------
        A list of phrases (spacy.tokens.span.Span) found in the doc
        """
        
        sent_tokens = [t.lemma_ if use_lemma and t.pos_ == 'NOUN' else t.text for t in doc]
        phrase_tokens = [t.lemma_ if use_lemma and t.pos_ == 'NOUN' else t.text for t in phrase]

        phrase_length = len(phrase_tokens)
        if lower:
            sent_tokens = [token.lower() for token in sent_tokens]
            phrase_tokens = [token.lower() for token in phrase_tokens]
        
        return [doc[i : i + phrase_length] for i in range(len(doc)-phrase_length+1) if phrase_tokens == sent_tokens[i : i + phrase_length]]
    
    
    modifier_dependencies = {'acl', 'advcl', 'advmod', 'amod', 'det', 'mark', 'meta', 'neg', 'nn', 'nmod', 'npmod', 'nummod', 'poss', 'prep', 'quantmod', 'relcl',
                             'appos', 'aux', 'auxpass', 'compound', 'cop', 'ccomp', 'xcomp', 'acomp', 'pcomp', 'expl', 'punct', 'nsubj', 'nsubjpass', 'csubj', 'csubjpass', 'agent', 'dobj', 'iobj', 'dative', 'obj', 'pobj', 'attr'}
    
    @staticmethod
    def form_info(doc:Doc, old_subjs:List[Tuple[int, int]], old_objs:List[Tuple[int, int]], new_subjs:List[Tuple[int, int]], new_objs:List[Tuple[int, int]]):

        new_subjs_idx, new_objs_idx, old_subjs_idx, old_objs_idx = 0, 0, 0, 0
        
        info:List[Tuple[Tuple[int, int], Tuple[int, int], str, List[str]]] = []
        punct_cnts = []
        for sent in doc.sents:
            punct_cnts.append(sum(t.pos_ == 'PUNCT' for t in sent))
            temp_new_subjs = []
            temp_new_objs = []
            temp_old_subjs = []
            temp_old_objs = []
            while new_subjs_idx < len(new_subjs) and new_subjs[new_subjs_idx][1] <= sent.end:
                temp_new_subjs.append(new_subjs[new_subjs_idx])
                new_subjs_idx += 1
            while new_objs_idx < len(new_objs) and new_objs[new_objs_idx][1] <= sent.end:
                temp_new_objs.append(new_objs[new_objs_idx])
                new_objs_idx += 1
            while old_subjs_idx < len(old_subjs) and old_subjs[old_subjs_idx][1] <= sent.end:
                temp_old_subjs.append(old_subjs[old_subjs_idx])
                old_subjs_idx += 1
            while old_objs_idx < len(old_objs) and old_objs[old_objs_idx][1] <= sent.end:
                temp_old_objs.append(old_objs[old_objs_idx])
                old_objs_idx += 1
            
            if temp_new_subjs:
                temp_new_subjs = [doc[span[0]:span[1]] for span in temp_new_subjs]
                
            if temp_new_objs:
                temp_new_objs = [doc[span[0]:span[1]] for span in temp_new_objs]
                
            if temp_new_subjs:
                temp_old_objs = [doc[span[0]:span[1]] for span in temp_old_objs]
                temp_objs = temp_new_objs + temp_old_objs
                if temp_objs:
                    info.extend([
                        ((subj.start, subj.end), (obj.start, obj.end), core_pattern, sub_patterns) 
                        for subj, obj, core_pattern, sub_patterns 
                        in SentenceAnalyzerBase.batch_collect_pattern(doc, product(
                            temp_new_subjs, temp_objs))
                    ])
            if temp_new_objs:
                temp_old_subjs = [doc[span[0]:span[1]] for span in temp_old_subjs]
                if temp_old_subjs:
                    info.extend([
                        ((subj.start, subj.end), (obj.start, obj.end), core_pattern, sub_patterns) 
                        for subj, obj, core_pattern, sub_patterns 
                        in SentenceAnalyzerBase.batch_collect_pattern(doc, product(temp_old_subjs, temp_new_objs))
                    ])
        return info
        
    # --------------------------------------------------------- Static functions end ---------------------------------------------------------
    
    # --------------------------------------------------------- Object functions start ---------------------------------------------------------
    def update_explicitness(self, update_core:bool=True, update_sub:bool=True):
        if update_core:
            self.core_log_max_cnt = np.log(self.corepath_counter.most_common(1)[0][1]+1)
            self.corepath_explicitness = defaultdict(lambda: np.log(1+.5) / self.core_log_max_cnt)
            for k, v in self.corepath_counter.items():
                self.corepath_explicitness[k] = np.log(v+1) / self.core_log_max_cnt
                
        if update_sub:
            self.sub_log_max_cnt = np.log(self.subpath_counter.most_common(1)[0][1]+1)
            self.subpath_explicitness = defaultdict(lambda: np.log(1+.5) / self.sub_log_max_cnt)
            for k, v in self.subpath_counter.items():
                self.subpath_explicitness[k] = np.log(v+1) / self.sub_log_max_cnt
    
    
    def get_path_cnt(self, data:List, original_core_path_generator:IdGenerator, original_sub_path_generator:IdGenerator):
        for p in tqdm(data):
            info:list = p['info']
            length = p['length']
            info.sort(key=lambda x: x[0][0])
            sent_id = 0
            sent_starts:list = p['sents']
            sent_ends = sent_starts[1:] + [length]
            sent_cores = defaultdict(set)
            sent_subs = defaultdict(set)
            item:list
            for item in info:
                # Replace path ids with path str
                subj_span, obj_span, core_path, sub_paths = item
                # Find the sentence which the current ent is in
                while subj_span[0] >= sent_ends[sent_id]:
                    sent_id += 1
                sent_cores[sent_id].add(self.corepath_id_generator.get_id(original_core_path_generator.get_obj(core_path)))
                sent_subs[sent_id].update([self.subpath_id_generator.get_id(original_sub_path_generator.get_obj(sub_path)) for sub_path in sub_paths])
                
            
            # Update core/sub path frequency
            for sent_id, cores in sent_cores.items():
                self.corepath_counter.update(cores)
            for sent_id, subs in sent_subs.items():
                self.subpath_counter.update(subs)
            
        self.update_explicitness()
        
    
    def save(self, corepath_file:str, subpath_file:str):
        self.corepath_id_generator.save(corepath_file)
        self.subpath_id_generator.save(subpath_file)


    def save_cnt(self, corepath_cnt_file:str, subpath_cnt_file:str):
        write_csv(corepath_cnt_file, [[str(k), str(v)] for k, v in self.corepath_counter.items()])
        write_csv(subpath_cnt_file, [[str(k), str(v)] for k, v in self.subpath_counter.items()])


    def cal_score(self, corepath:int, subpaths:List[int], sent_length:int, ent_length:int, punct_num:int=0):
        s:float = min((sum([self.subpath_explicitness[p] for p in subpaths]) + ent_length) / (sent_length - punct_num), 1)
        e:float = self.corepath_explicitness[corepath]
        return {'score' : 2 / ((1/s)+(1/e)), 'significance' : s, 'explicitness' : e}
    
    
    def informativeness_demo(self, doc:Doc, kw1_span:Span, kw2_span:Span):
        kw1_steps, kw2_steps, branch = SentenceAnalyzerBase.find_dependency_info_from_tree(doc, kw1_span, kw2_span)
        self.expand_dependency_info_from_tree(doc, branch)
        return pd.DataFrame({i:[doc[i].text, np.round(branch[i], 3)] for i in range(len(doc))})


    def expand_dependency_info_from_tree(self, doc:Doc, branch:np.ndarray):
        dep_path:List[int] = np.arange(*branch.shape)[branch!=0]
        for element in dep_path:
            if doc[element].dep_ == 'conj':
                branch[doc[element].head.i] = 0
        paths = SentenceAnalyzerBase.collect_sub_dependency_path(doc, branch)
        for p in paths:
            pattern = SentenceAnalyzerBase.gen_subpath_pattern(p[1])
            if pattern == '':
                branch[p[2]] = branch[p[0]]
            else:
                branch[p[2]] = self.subpath_explicitness[self.subpath_id_generator.get_id(pattern)]
                
    
    def process(self, passages:List[dict], titles:List[str], dump_file:str=None, show_status:bool=False, return_doc:bool=False, n_process:int=10, skip_empty:bool=True, work_name:str=''):
        paragraphs = [p['text'] for p in passages]
        try:
            ii = 0
            paragraph_doc = self.spacy_nlp.pipe(paragraphs, n_process=n_process)
            if show_status:
                docs = tqdm(paragraph_doc, total=len(paragraphs))
            else:
                docs = paragraph_doc
            data = []
            for ii, (doc, title, p) in enumerate(zip(docs, titles, passages)):
                span2ontologies = self.get_span2ontologies_from_doc(doc)
                info:List[Tuple[Tuple[int, int], Tuple[int, int], str, List[str]]] = doc._.patterns
                old_subjs:List[Tuple[int, int]] = doc._.subjs
                old_objs:List[Tuple[int, int]] = doc._.objs
                
                new_subjs:List[Tuple[int, int]] = []
                new_objs:List[Tuple[int, int]] = []
                self.add_additional_entities(p, doc, span2ontologies, new_subjs, new_objs)
                
                if not span2ontologies and skip_empty:
                    continue
                
                cooccurs = list(span2ontologies.items())
                cooccurs.sort(key=lambda x: x[0][0])
                
                if new_subjs or new_objs:
                    new_subjs.sort(key=lambda x: x[0])
                    new_objs.sort(key=lambda x: x[0])
                    info.extend(self.extend_info(doc, old_subjs, old_objs, new_subjs, new_objs))
                
                data.append({'text' : doc.text, 
                             'title' : title, 
                             'info' : [(subj, obj, self.corepath_id_generator.get_id(corepath), [self.subpath_id_generator.get_id(p) for p in subpaths]) for subj, obj, corepath, subpaths in info], 
                             'cooccurs' : cooccurs, 
                             'sents' : [sent[0].i for sent in doc.sents], 
                             'punct_cnts' : [sum(t.pos_ == 'PUNCT' for t in sent) for sent in doc.sents], 
                             'length' : len(doc)})
                if return_doc:
                    data[-1]['doc'] = doc
        
        except:
            with open('error.log', 'a') as f_out:
                f_out.write('%s %s\n' % (str(ii), work_name))
        finally:
            if dump_file:
                write_json(dump_file, data)
            else:
                return data
        
    # --------------------------------------------------------- Object functions end ---------------------------------------------------------
    
    # --------------------------------------------------------- Virtual functions start ---------------------------------------------------------
    def get_span2ontologies_from_doc(self, doc:Doc) -> Dict[Tuple[int, int], Dict[str, List[Tuple[str, float]]]]:
        '''
        Return Dict[entity_span, Dict[ontology_name, List[Tuple[entity_id, confidence_score, (entity_name)]]]]
        '''
        raise NotImplementedError
    
    
    def add_additional_entities(self, p:Dict[str, Any], doc:Doc, span2ontologies:Dict[Tuple[int, int], Dict[str, List[Tuple[str, float]]]], new_subjs:List[Tuple[int, int]], new_objs:List[Tuple[int, int]]) -> None:
        raise NotImplementedError
    
    
    def extend_info(self, doc:Doc, old_subjs:List[Tuple[int, int]], old_objs:List[Tuple[int, int]], new_subjs:List[Tuple[int, int]], new_objs:List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int], str, List[str]]]:
        raise NotImplementedError
    
    # --------------------------------------------------------- Virtual functions end ---------------------------------------------------------
    
    
@Language.factory("scispacy_multi_linker")
class MultiEntityLinker:
    def __init__(
        self,
        nlp: Language = None,
        name: str = "scispacy_multi_linker",
        resolve_abbreviations: bool = True,
        k: int = 30,
        threshold: float = 0.8,
        no_definition_threshold: float = 0.95,
        filter_for_definitions: bool = True,
        max_entities_per_mention: int = 3,
        linker_names: List[str] = ['umls'],
    ):
        Span.set_extension('ontologies', default={}, force=True)
        
        self.candidate_generators:Dict[str, CandidateGenerator] = {}
        if 'umls' in linker_names:
            self.candidate_generators['umls'] = CandidateGenerator(name = 'umls')
        if 'mesh' in linker_names:
            self.candidate_generators['mesh'] = CandidateGenerator(name = 'mesh')
        if 'rxnorm' in linker_names:
            self.candidate_generators['rxnorm'] = CandidateGenerator(name = 'rxnorm')
        if 'go' in linker_names:
            self.candidate_generators['go'] = CandidateGenerator(name = 'go')
        if 'hpo' in linker_names:
            self.candidate_generators['hpo'] = CandidateGenerator(name = 'hpo')
        
        self.resolve_abbreviations = resolve_abbreviations
        self.k = k
        self.threshold = threshold
        self.no_definition_threshold = no_definition_threshold
        self.filter_for_definitions = filter_for_definitions
        self.max_entities_per_mention = max_entities_per_mention

    def __call__(self, doc: Doc) -> Doc:
        mention_strings = []
        splits = []
        if self.resolve_abbreviations and Doc.has_extension("abbreviations"):
            # TODO: This is possibly sub-optimal - we might
            # prefer to look up both the long and short forms.
            for ent in doc.ents:
                if isinstance(ent._.long_form, Span):
                    # Long form
                    mention_strings.append(ent._.long_form.text.lower())
                    splits.append(1)
                elif isinstance(ent._.long_form, str):
                    # Long form
                    mention_strings.append(ent._.long_form.lower())
                    splits.append(1)
                else:
                    # no abbreviations case
                    original_text = ent.text.lower()
                    mention_strings.append(original_text)
                    splits.append(1)
                    lemmazied_text = ' '.join([t.lemma_.lower() if t.pos == 92 else t.text.lower() for t in ent])
                    if original_text != lemmazied_text:
                        mention_strings.append(lemmazied_text)
                        splits[-1] += 1
        else:
            mention_strings = [[' '.join([t.lemma_ if t.pos == 92 else t.text for t in ent]), ent.text] for ent in doc.ents]

        for mention in doc.ents:
            mention._.ontologies = {}
            
        for ontology_name, candidate_generator in self.candidate_generators.items():
            batch_candidates = candidate_generator(mention_strings, self.k)
            ent_idx = 0
            for mention, split in zip(doc.ents, splits):
                candidates = batch_candidates[ent_idx] + batch_candidates[ent_idx+1] if split == 2 else batch_candidates[ent_idx]
                temp_mention_strings = mention_strings[ent_idx:ent_idx+split]
                ent_idx += split
                predicted = set()
                for cand in candidates:
                    score = max(cand.similarities)
                    if (
                        self.filter_for_definitions
                        and candidate_generator.kb.cui_to_entity[cand.concept_id].definition is None
                        and score < self.no_definition_threshold
                    ):
                        continue
                    if score > self.threshold:
                        aliases = [alias.lower() for alias in cand.aliases]
                        exact_match = False
                        for temp_mention in temp_mention_strings:
                            exact_match = (temp_mention in aliases) or exact_match
                        predicted.add((cand.concept_id, score, exact_match))
                if not predicted:
                    continue
                exact_match_exist = any([pred[2] for pred in predicted])
                predicted = [pred[:2] for pred in predicted if pred[2] or (not exact_match_exist)]
                predicted.sort(key=lambda x: x[1], reverse=True)
                predicted = predicted[: self.max_entities_per_mention]
                mention._.ontologies[ontology_name] = predicted

        return doc
    

from spacy_entity_linker.EntityClassifier import EntityClassifier
from spacy_entity_linker.TermCandidateExtractor import TermCandidateExtractor

@Language.factory('spacyEntityLinker')
class SpacyEntityLinker:

    def __init__(self, nlp, name):
        Doc.set_extension("linkedEntities", default=None, force=True)

    def __call__(self, doc):
        tce = TermCandidateExtractor(doc)
        classifier = EntityClassifier()

        entities = []
        for termCandidates in tce:
            entityCandidates = termCandidates.get_entity_candidates()
            if len(entityCandidates) > 0:
                entity = classifier(entityCandidates)
                entities.append(entity)

        doc._.linkedEntities = entities

        return doc
    

def is_numbers(s:str):
    s = s.replace('.', '', 1).strip('%')
    return s.isnumeric()


@Language.factory("pattern_extractor")
class PatternExtractor:
    def __init__(
        self,
        nlp: Language = None,
        name: str = "pattern_extractor"
    ):
        Doc.set_extension("patterns", default=[], force=True)
        Doc.set_extension("subjs", default=[], force=True)
        Doc.set_extension("objs", default=[], force=True)

    def __call__(self, doc: Doc) -> Doc:
        cooccurs:List[Span] = []
        if Span.has_extension('ontologies'):
            # Process entities extracted by Scispacy's entity linker
            for ent in doc.ents:
                if ent.root.pos_ != 'NOUN':
                    continue
                if not ent._.ontologies:
                    continue
                cooccurs.append(ent)
        elif Doc.has_extension('linkedEntities'):
            # Process entities extracted by Spacy's entity linker
            linkedEntities:List[EntityElement] = doc._.linkedEntities
            cooccurs = [ent.get_span() for ent in linkedEntities]
            doc._.linkedEntities = [((ent.get_span().start, ent.get_span().end), ent.get_id(), ent.get_label()) for ent in linkedEntities]
        else:
            # Process entities and noun phrases extracted by base pipeline
            mask = np.zeros(len(doc))
            ent_id = 1
            for ent in doc.ents:
                mask[ent.start:ent.end] = ent_id
                ent_id += 1
            for ent in doc.noun_chunks:
                if mask[ent.start] == 0 and mask[ent.end-1] == 0:
                    mask[ent.start:ent.end] = ent_id
                    ent_id += 1
                else:
                    mask[ent.start:ent.end] = max(mask[ent.start], mask[ent.end-1])
            b = np.arange(len(mask)-1)
            start = 0
            for end in (b[mask[b] != mask[b+1]]+1).tolist() + [len(mask)]:
                if (mask[start:end] > 0).all():
                    temp_span = doc[start:end]
                    if temp_span.root.pos_ in {'NOUN', 'PROPN'}:
                        cooccurs.append(temp_span)
                start = end
            
        cooccurs.sort(key=lambda x: x.start)
        
        info:List[Tuple[Tuple[int, int], Tuple[int, int], str, List[str]]] = []
        subj_list:List[Tuple[int, int]] = []
        obj_list:List[Tuple[int, int]] = []
        cooccur_idx = 0
        for sent in doc.sents:
            temp_ents:List[Span] = []
            while cooccur_idx < len(cooccurs) and cooccurs[cooccur_idx].end <= sent.end:
                temp_ents.append(cooccurs[cooccur_idx])
                cooccur_idx += 1
            if len(temp_ents) <= 1:
                continue
            temp_subj_list:List[Span] = []
            temp_obj_list:List[Span] = []
            for span in temp_ents:
                if SentenceAnalyzerBase.is_subj(span):
                    temp_subj_list.append(span)
                else:
                    temp_obj_list.append(span)
            subj_list.extend([(s.start, s.end) for s in temp_subj_list])
            obj_list.extend([(o.start, o.end) for o in temp_obj_list])
            if not temp_subj_list or not temp_obj_list:
                continue
            info.extend([
                ((subj.start, subj.end), (obj.start, obj.end), core_pattern, sub_patterns) 
                for subj, obj, core_pattern, sub_patterns 
                in SentenceAnalyzerBase.batch_collect_pattern(doc, product(temp_subj_list, temp_obj_list))
            ])
        doc._.patterns = info
        doc._.subjs = subj_list
        doc._.objs = obj_list
            
        return doc


class RelationDescriptionEmbedding:
    def __init__(self, transformer_tokenizer:BertTokenizer, nlp:Language, transformer_encoder:BertModel):
        self.transformer_tokenizer = transformer_tokenizer
        self.nlp = nlp
        self.transformer_encoder = transformer_encoder

    def group_sub_words(self, word_ids:List[int], input_ids):
        # Group sub-words
        current_word = -1
        start_idx = 0
        spans = []
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                if current_word < 0:
                    continue
                else:
                    spans.append((self.transformer_tokenizer.decode(input_ids[start_idx:idx]), (start_idx, idx)))
                    break
            elif word_id > current_word:
                if current_word >= 0:
                    spans.append((self.transformer_tokenizer.decode(input_ids[start_idx:idx]), (start_idx, idx)))
                current_word = word_id
                start_idx = idx
        return spans
    
    def align_transformer_spacy_seqs(self, spans:List[Tuple[str, Tuple[int, int]]], spacy_tokenized:List[str]):
        spacy_start = 0
        spacy_end = 1
        token_buffer = []
        span_buffer = []
        span_spacy_span_pairs = []
        for word, span in spans:
            if word == ''.join(spacy_tokenized[spacy_start:spacy_end]):
                span_spacy_span_pairs.append((span, (spacy_start, spacy_end)))
                spacy_start = spacy_end
                spacy_end += 1
            else:
                token_buffer.append(word)
                span_buffer.append(span)
                temp_tokens_length = sum([len(temp_token) for temp_token in token_buffer])
                while temp_tokens_length > sum([len(spacy_tokenized[spacy_idx]) for spacy_idx in range(spacy_start, spacy_end)]):
                    spacy_end += 1
                if temp_tokens_length == sum([len(spacy_tokenized[spacy_idx]) for spacy_idx in range(spacy_start, spacy_end)]):
                    span_spacy_span_pairs.append(((span_buffer[0][0], span_buffer[-1][-1]), (spacy_start, spacy_end)))
                    spacy_start = spacy_end
                    spacy_end += 1
                    token_buffer = []
                    span_buffer = []
                
        if token_buffer:
            span_spacy_span_pairs.append(((span_buffer[0][0], span_buffer[-1][-1]), (spacy_start, spacy_end)))
        
        if span_spacy_span_pairs[-1][1][1] != len(spacy_tokenized):
            print('')
            
        assert span_spacy_span_pairs[-1][1][1] == len(spacy_tokenized)
        
        return span_spacy_span_pairs
    
    def map_weight(self, span_spacy_span_pairs:List[Tuple[Tuple[int, int], Tuple[int, int]]], score:List[float], size:int):
        weights = torch.zeros(size)
        for span, spacy_span in span_spacy_span_pairs:
            weights[span[0]:span[1]] = mean(score[spacy_span[0]:spacy_span[1]])
        return weights
    
    def get_weights(self, encoded:BatchEncoding, spacy_tokenized:List[str], score:List[float]):
        input_ids:torch.LongTensor = encoded.input_ids
        spans = self.group_sub_words(encoded.word_ids(), input_ids)
        span_spacy_span_pairs = self.align_transformer_spacy_seqs(spans, spacy_tokenized)
        return self.map_weight(span_spacy_span_pairs, score, len(input_ids))
    
    def batched_get_weights(self, encoded:BatchEncoding, spacy_tokenized_list:List[List[str]], scores:List[List[float]], mask_spans:List[List[Tuple[int, int]]]=None):
        input_ids:torch.LongTensor = encoded.input_ids
        return torch.vstack([
            self.map_weight(
                self.align_transformer_spacy_seqs(
                    self.group_sub_words(encoded.word_ids(i), input_ids[i]), 
                    spacy_tokenized), score, input_ids.shape[1])
            for i, (spacy_tokenized, score) in enumerate(zip(spacy_tokenized_list, scores))
        ])
    
    @staticmethod
    def softmax_weight(weights:torch.FloatTensor):
        return torch.softmax(weights - torch.where(weights > 0, torch.zeros_like(weights), torch.ones_like(weights) * float('inf')), dim=-1)
    
    def generate_embedding(self, sents:List[str], scores:List[List[float]], mask_spans:List[List[Tuple[int, int]]]=None, batch_size:int=500):
        spacy_tokenized_list = [[t.text for t in self.nlp.tokenizer(sent)] for sent in sents]
        if mask_spans is not None:
            for i, (spacy_tokenized, score, mask_span) in enumerate(zip(spacy_tokenized_list, scores, mask_spans)):
                mask_span.sort(key=lambda x:x[0])
                offset = 0
                for span in mask_span:
                    spacy_tokenized = spacy_tokenized[:span[0]-offset] + [self.transformer_tokenizer.mask_token] + spacy_tokenized[span[1]-offset:]
                    score = score[:span[0]-offset] + [mean(score[span[0]-offset : span[1]-offset])] + score[span[1]-offset:]
                    offset += span[1] - span[0] - 1
                spacy_tokenized_list[i] = spacy_tokenized
                scores[i] = score
                sents[i] = ' '.join(spacy_tokenized)
            
        with torch.no_grad():
            steps = ((len(sents) - 1) // batch_size) + 1
            emb_list = []
            for step in tqdm(range(steps)):
                encoded = self.transformer_tokenizer.batch_encode_plus(sents[step*batch_size : (step+1)*batch_size], max_length=512, padding=True, return_tensors='pt', truncation=True)
                last_hidden_state:torch.FloatTensor = self.transformer_encoder(**(encoded.to(self.transformer_encoder.device)))[0]
                weights = self.batched_get_weights(encoded, spacy_tokenized_list[step*batch_size : (step+1)*batch_size], scores[step*batch_size : (step+1)*batch_size]).to(self.transformer_encoder.device)
                weights = RelationDescriptionEmbedding.softmax_weight(weights)
                emb = torch.matmul(weights.unsqueeze(1), last_hidden_state).squeeze(1).cpu()
                emb_list.append(emb)
            return torch.vstack(emb_list)
        