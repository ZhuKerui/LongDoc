# from .base import *
from spacy.tokens import Token, Span, Doc

'''
conj, appos: follow the dep of the head
compound, nmod, amod, det, poss: merge
npadvmod, ROOT, cc, dep, pcomp: miss
'''

class SentenceAnalyzerBase:
            
    # --------------------------------------------------------- Static functions start ---------------------------------------------------------
    @staticmethod
    def get_full_noun_phrase(noun_phrase:Span):
        # Get the root of the noun phrase
        temp_front_token = noun_phrase[0]
        temp_back_token = noun_phrase[-1]
        noun_phrase_deps = {'compound', 'nmod', 'amod', 'det', 'poss'}
        
        root = noun_phrase.root
        temp_front_token = root if temp_front_token.i > root.i else temp_front_token
        temp_back_token = root if temp_back_token.i < root.i else temp_back_token
        while root.dep_ in noun_phrase_deps:
            root = root.head
            temp_front_token = root if temp_front_token.i > root.i else temp_front_token
            temp_back_token = root if temp_back_token.i < root.i else temp_back_token
        
        childrens = [child for child in root.children if child.dep_ in noun_phrase_deps]
        while childrens:
            new_childrens = []
            for c in childrens:
                temp_front_token = c if temp_front_token.i > c.i else temp_front_token
                temp_back_token = c if temp_back_token.i < c.i else temp_back_token
                new_childrens.extend(child for child in c.children if child.dep_ in noun_phrase_deps)
            childrens = new_childrens
        return root.doc[temp_front_token.i:temp_back_token.i+1]

