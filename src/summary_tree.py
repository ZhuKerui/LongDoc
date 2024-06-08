from .data import *
from .prompt import LongDocPrompt
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace
from langchain_text_splitters import SpacyTextSplitter
from langchain_openai import ChatOpenAI

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain import hub

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
    



class TreeIndex(MyIndex):
    is_leaf: bool
    level: int
    level_index: int
    summary:str = ''
    children:List[int] = []
    parent:List[int] = []
    

class TreeNode(MyNode):
    index:TreeIndex
    
            
class MyTree(MyStructure):
    node_class = TreeNode
    name = 'tree'
    
    summary_by_chunks_prompt = (
        "The following passage is a concatenation of chunks.\n"
        "Use all the chunks to help you understand the general context.\n"
        "Summarize each chunk seperately so that all the chunk summaries together will form a coherent summary for the passage.\n"
        "Each chunk summary should start with the corresponding chunk header, like 'Chunk 1: [summary for chunk 1]'.\n"
        "\n"
        "\n"
        "{context_str}\n"
        "\n"
        "\n"
        'SUMMARY:"""\n'
    )
    
    def __init__(self, doc_file: str = None) -> None:
        self.tree:List[List[int]] = []
        self.docs:List[TreeNode] = []
        super().__init__(doc_file)
    
    def load(self, doc_file: str):
        super().load(doc_file)
        level_node_cnt = Counter([doc.index.level for doc in self.docs])
        self.tree = [[0] * level_node_cnt[level] for level in sorted(level_node_cnt.keys())]
        for doc in self.docs:
            self.tree[doc.index.level][doc.index.level_index] = doc.index.i
        
    @property
    def height(self):
        return len(self.tree)
    
    def __getitem__(self, i:int):
        return self.tree[i]
    
    def append_layer(self, level:List[TreeNode]):
        self.docs.extend(level)
        self.tree.append([n.index.i for n in level])
        
    def retrieve_branches(self, node:TreeNode):
        branches:List[List[int]] = [[node.index.i]]
        while self.docs[branches[0][-1]].index.parent:
            new_branches:List[List[int]] = []
            for branch in branches:
                for p in self.docs[branch[-1]].index.parent:
                    new_branches.append(branch + [p])
            branches = new_branches
        return branches
    
    @classmethod
    def build_summary_pyramid(
        cls,
        llm:BaseChatModel, 
        pages:List[str],
        chunk_num:int = 3
    ):
        tree = cls()
        prompts_check = []
        while tree.height == 0 or len(tree[-1]) > 1:
            current_level_nodes:List[TreeNode] = []
            if tree.height == 0:
                current_level_nodes = [TreeNode(text=doc, index=TreeIndex(is_leaf=True, level=tree.height, i=Doc_id, level_index=Doc_id)) for Doc_id, doc in enumerate(pages)]
            else:
                prev_level = tree[-1]
                prev_level_length = len(prev_level)
                if prev_level_length % chunk_num == 0:
                    chunk_distributions = [chunk_num] * (prev_level_length // chunk_num)
                else:
                    sub_chunk_num = chunk_num - 1
                    sub_chunk_cnt = chunk_num - (prev_level_length % chunk_num)
                    chunk_distributions = ([sub_chunk_num] * sub_chunk_cnt) + ([chunk_num] * ((prev_level_length - sub_chunk_num * sub_chunk_cnt) // chunk_num))
                    random.shuffle(chunk_distributions)
                
                batch_start = 0
                for bid, batch_size in enumerate(chunk_distributions):
                    batch_end = batch_start + batch_size
                    current_node_index = TreeIndex(is_leaf=False, level=tree.height, i=bid + len(tree.docs), level_index=bid)
                    for child in prev_level[batch_start : batch_end]:
                        current_node_index.children.append(child)
                        tree.docs[child].index.parent.append(current_node_index.i)
                    current_level_nodes.append(TreeNode(text='', index=current_node_index))
                    batch_start = batch_end
                
                current_passages = ['\n'.join([f'Chunk {child}: {tree.docs[child].text}' for child in node.index.children]) for node in current_level_nodes]
                summary_prompts = [[HumanMessage(content=cls.summary_by_chunks_prompt.format(context_str=current_passage))] for current_passage in current_passages]
                prompts_check.append(summary_prompts)
                print('Summarize:', len(summary_prompts))
                summaries = [response[0].text for response in llm.generate(summary_prompts).generations]
                for node in current_level_nodes:
                    node.index.summary = summaries[node.index.level_index]
                    node.text = node.index.summary
                    for child in node.index.children:
                        node.text = node.text.replace(f'Chunk {child}:', '')
                    node.text = ' '.join(node.text.split())
            tree.append_layer(current_level_nodes)
        return tree, prompts_check


class DPRIndex(MyIndex):
    pass
    
    
class DPRNode(MyNode):
    index:DPRIndex
    
    
class MyDPR(MyStructure):
    node_class = DPRNode
    name = 'dpr'
    
    def __init__(self, doc_file: str = None) -> None:
        self.docs:List[DPRNode] = []
        super().__init__(doc_file)
    
    @classmethod
    def build_dpr(cls, pages:List[str]):
        dpr = cls()
        dpr.docs = [DPRNode(text=doc, index=DPRIndex(i=doc_id)) for doc_id, doc in enumerate(pages)]
        return dpr
    



class GradeDocument:
    
    class GradeDocumentOutput(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, yes or no"
        )
        
    prompt_wo_context_str = LongDocPrompt.remove_multi_ws("""
        You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        {format_instructions}
        
        Retrieved document:
        {document}
        
        User question:
        {question}
    """)
    
    prompt_w_context_str = LongDocPrompt.remove_multi_ws("""
        You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        A context of known information is also provided. You should make use of the context to help you connect the retrieved document with the question.
        {format_instructions}
        
        Context:
        {context}
        
        Retrieved document:
        {document}
        
        User question:
        {question}
    """)
    
    def __init__(self, llm: BaseChatModel) -> None:
        output_parser = PydanticOutputParser(pydantic_object=self.GradeDocumentOutput)
        prompt_wo_context = PromptTemplate(
            template=self.prompt_wo_context_str,
            input_variables=['document', 'question'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        prompt_w_context = PromptTemplate(
            template=self.prompt_w_context_str,
            input_variables=['document', 'question', 'context'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        self.chain_wo_context = prompt_wo_context | llm | output_parser
        self.chain_w_context = prompt_w_context | llm | output_parser
        
    def __call__(self, documents:List[str], questions:List[str], contexts:List[str]=None) -> List[GradeDocumentOutput]:
        if contexts is None:
            return self.chain_wo_context.batch([{'question': question, 'document': doc} for doc, question in zip(documents, questions)])
        else:
            return self.chain_w_context.batch([{'question': question, 'document': doc, 'context': ctx} for doc, question, ctx in zip(documents, questions, contexts)])


class AnalyzeDocument:
    
    prompt_wo_context_str = LongDocPrompt.remove_multi_ws("""
        You are an information extractor.
        Extract the information in a given document that is useful for answer the user question.
        
        Document:
        {document}
        
        User question:
        {question}
        
        Useful Information:
    """)
    
    prompt_w_context_str = LongDocPrompt.remove_multi_ws("""
        You are an information extractor.
        Extract the information in a given document that is useful for answer the user question.
        A context of known information is also provided. You should make use of the context to help you connect the document with the question.
        
        Context:
        {context}
        
        Document:
        {document}
        
        User question:
        {question}
        
        Useful Information:
    """)
    
    def __init__(self, llm: BaseChatModel) -> None:
        output_parser = StrOutputParser()
        prompt_wo_context = PromptTemplate(
            template=self.prompt_wo_context_str,
            input_variables=['document', 'question'],
        )
        prompt_w_context = PromptTemplate(
            template=self.prompt_w_context_str,
            input_variables=['document', 'question', 'context'],
        )
        self.chain_wo_context = prompt_wo_context | llm | output_parser
        self.chain_w_context = prompt_w_context | llm | output_parser
        
    def __call__(self, documents:List[str], questions:List[str], contexts:List[str]=None):
        if contexts is None:
            return self.chain_wo_context.batch([{'question': question, 'document': doc} for doc, question in zip(documents, questions)])
        else:
            return self.chain_w_context.batch([{'question': question, 'document': doc, 'context': ctx} for doc, question, ctx in zip(documents, questions, contexts)])
    
        
class GenerateAnswer:
    
    prompt:ChatPromptTemplate = hub.pull("rlm/rag-prompt")
    
    def __init__(self, llm: BaseChatModel) -> None:
        self.chain = self.prompt | llm | StrOutputParser()
        
    def __call__(self, contexts:List[str], questions:List[str]):
        return self.chain.batch([{'question': question, 'context': context} for context, question in zip(contexts, questions)])
        
        
class EvalCompleteInfo:
    
    class EvalCompleteInfoOutput(BaseModel):
        """Binary score for complete information check on context."""

        binary_score: str = Field(
            description="The context contains complete information to answer the question, yes or no"
        )
        
    prompt_str = LongDocPrompt.remove_multi_ws("""
        You are a grader assessing the completeness of a context to answer a user question.
        If the context contains enough information to support a confident answer for the question, grade it as complete.
        It need to be a stringent evaluation, as the next step is to generate the final answer for the question. The goal is to avoid any missing information.
        Give a binary score 'yes' or 'no' score to indicate whether the context is complete to the question.
        {format_instructions}
        
        Context:
        {context}
        
        User question:
        {question}
    """)
    
    def __init__(self, llm: BaseChatModel) -> None:
        output_parser = PydanticOutputParser(pydantic_object=self.EvalCompleteInfoOutput)
        prompt = PromptTemplate(
            template=self.prompt_str,
            input_variables=['context', 'question'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        self.chain = prompt | llm | output_parser
        
    def __call__(self, contexts:List[str], questions:List[str]) -> List[EvalCompleteInfoOutput]:
        return self.chain.batch([{'question': question, 'context': context} for context, question in zip(contexts, questions)])


class GenerateQuery:
    
    class GenerateQueryOutput(BaseModel):
        """Queries for retrieving information relevant to the question."""
    
        queries: List[str] = Field(
            description='A list of queries'
        )
        
    prompt_w_context_str = LongDocPrompt.remove_multi_ws("""
        Write queries to retrieve information relevant to a user question.
        You may generate a single query or multiple queries, depending on the information you need.
        A context of known information is also provided. You should make use of the context and generate queries for the missing information.
        {format_instructions}
        
        Context:
        {context}
        
        User question:
        {question}
    """)
    
    prompt_wo_context_str = LongDocPrompt.remove_multi_ws("""
        Write queries to retrieve information relevant to a user question.
        You may generate a single query or multiple queries, depending on the information you need.
        {format_instructions}
        
        User question:
        {question}
    """)
    
    def __init__(self, llm:BaseChatModel) -> None:
        output_parser = PydanticOutputParser(pydantic_object=self.GenerateQueryOutput)
        prompt_w_context = PromptTemplate(
            template=self.prompt_wo_context_str,
            input_variables=['context', 'question'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        
        prompt_wo_context = PromptTemplate(
            template=self.prompt_wo_context_str,
            input_variables=['question'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        self.chain_w_context = prompt_w_context | llm | output_parser
        self.chain_wo_context = prompt_wo_context | llm | output_parser
        
    def __call__(self, questions:List[str], contexts:List[str]=None) -> List[GenerateQueryOutput]:
        if contexts is not None:
            return self.chain_w_context.batch([{'question': question, 'context': context} for context, question in zip(contexts, questions)])
        else:
            return self.chain_wo_context.batch([{'question': question} for question in questions])


class Factory:
    def __init__(self, embeder_name:str=None, llm_name:str = 'mistralai/Mistral-7B-Instruct-v0.2', chunk_size:int=300, device:str='cpu') -> None:
        if embeder_name is not None:
            self.embeder = HuggingFaceEmbeddings(model_name=embeder_name, model_kwargs={'device': device})
        else:
            self.embeder = HuggingFaceEmbeddings(model_kwargs={'device': device})
        self.embeder_name = self.embeder.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.embeder_name)
        self.splitter = SpacyTextSplitter(pipeline='en_core_web_lg', chunk_size=chunk_size, chunk_overlap=0, length_function=lambda x: len(self.tokenizer.encode(x, add_special_tokens=False)))
        
        
        self.llm_name = llm_name
        self.llm = ChatOpenAI(model=llm_name, base_url='http://128.174.136.28:8000/v1', temperature=0)
        
    def split_text(self, text:str):
        return [' '.join(t.split()) for t in self.splitter.split_text(text)]
    
    def build_corpus(self, text:str, dpr_file:str='temp_dpr.json', tree_file:str='temp_tree.json'):
        if not os.path.exists(dpr_file) or not os.path.exists(tree_file):
            pages = self.split_text(text)
        if os.path.exists(dpr_file):
            dpr_corpus = MyDPR(dpr_file)
        else:
            dpr_corpus = MyDPR.build_dpr(pages)
            dpr_corpus.dump(dpr_file)
        dpr_corpus.create_vector_retriever(self.embeder)
            
        if os.path.exists(tree_file):
            tree_corpus = MyTree(tree_file)
        else:
            tree_corpus, _ = MyTree.build_summary_pyramid(self.llm, pages, 3)
            tree_corpus.dump(tree_file)
        tree_corpus.create_vector_retriever(self.embeder)
        
        return {'dpr': dpr_corpus, 'tree': tree_corpus}
    
