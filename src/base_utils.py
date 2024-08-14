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

class MyIndex(BaseModel):
    i: int
    
class MyNode(BaseModel):
    text:str = ''
    index:MyIndex
    
    def to_doc(self):
        return Document(page_content=self.text, metadata={'i' : self.index.i})
    
class MyStructure:
    node_class = MyNode
    name:str
    
    def __init__(self, doc_file:str=None) -> None:
        self.docs:List[MyNode] = []
        if doc_file is not None:
            self.load(doc_file)
            
    def dump(self, doc_file:str):
        dumped_docs = [node.dict() for node in self.docs]
        write_json(doc_file, dumped_docs)
        
    def load(self, doc_file:str):
        dumped_tree = read_json(doc_file)
        self.docs = [self.node_class.validate(node_info) for node_info in dumped_tree]
        
    def create_vector_retriever(self, embedding:Embeddings):
        self.vectorstore = Chroma.from_documents(
            documents=[doc.to_doc() for doc in self.docs],
            collection_name=f"{self.name}-chroma",
            embedding=embedding,
        )
        self.retriever = self.vectorstore.as_retriever()
        
class Factory:
    def __init__(self, embeder_name:str=None, llm_name:str = DEFAULT_LLM, chunk_size:int=300, device:str='cpu') -> None:
        if embeder_name is not None:
            self.embeder = HuggingFaceEmbeddings(model_name=embeder_name, model_kwargs={'device': device})
        else:
            self.embeder = HuggingFaceEmbeddings(model_kwargs={'device': device})
        self.embeder_name = self.embeder.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.embeder_name)
        self.splitter = SpacyTextSplitter(pipeline='en_core_web_lg', chunk_size=chunk_size, chunk_overlap=0, length_function=lambda x: len(self.tokenizer.encode(x, add_special_tokens=False)))
        self.rouge = evaluate.load('rouge')
        
        
        self.llm_name = llm_name
        if self.llm_name:
            self.llm = ChatOpenAI(model=llm_name, base_url='http://128.174.136.28:8001/v1', temperature=0)
        
    def split_text(self, text:str):
        # return [' '.join(t.split()) for t in self.splitter.split_text(text)]
        return self.splitter.split_text(text)
    
    # def build_corpus(self, text:str, dpr_file:str='temp_dpr.json', tree_file:str='temp_tree.json'):
    #     if not os.path.exists(dpr_file) or not os.path.exists(tree_file):
    #         pages = self.split_text(text)
    #     if os.path.exists(dpr_file):
    #         dpr_corpus = MyDPR(dpr_file)
    #     else:
    #         dpr_corpus = MyDPR.build_dpr(pages)
    #         dpr_corpus.dump(dpr_file)
    #     dpr_corpus.create_vector_retriever(self.embeder)
            
    #     if os.path.exists(tree_file):
    #         tree_corpus = MyTree(tree_file)
    #     else:
    #         tree_corpus, _ = MyTree.build_summary_pyramid(self.llm, pages, 3)
    #         tree_corpus.dump(tree_file)
    #     tree_corpus.create_vector_retriever(self.embeder)
        
    #     return dpr_corpus, tree_corpus, [doc.to_doc() for doc in dpr_corpus.docs]
    
