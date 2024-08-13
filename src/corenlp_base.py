from .base import BaseModel, List, Tuple, Dict

class Token(BaseModel):
    index: int
    word: str
    originalText: str
    lemma: str
    characterOffsetBegin: int
    characterOffsetEnd: int
    pos: str
    ner: str
    speaker: str
    before: str
    after: str
    
    
class Dependency(BaseModel):
    dep: str
    governor: int
    governorGloss: str
    dependent: int
    dependentGloss: str
    
    
class Mention(BaseModel):
    id: int
    text: str
    type: str
    number: str
    startIndex: int
    endIndex: int
    headIndex: int
    sentNum: int
    position: Tuple[int, int]
    isRepresentativeMention: bool
    
    
class Sentence(BaseModel):
    index: int
    tokens: List[Token]
    basicDependencies: List[Dependency]
    enhancedDependencies: List[Dependency]
    enhancedPlusPlusDependencies: List[Dependency]
    
    
class Doc(BaseModel):
    sentences: List[Sentence]
    corefs: Dict[str, List[Mention]]
    
    