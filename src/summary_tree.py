from .data import *
from .prompt import LongDocPrompt

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
    
    def retrieve_children(self, node:TreeNode):
        children:List[int] = [node.index.i]
        while children[0] not in self.tree[0]:
            temp_children = set()
            for child in children:
                temp_children.update(self.docs[child].index.children)
            children = list(temp_children)
        return children
            
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
        """Score for relevance check on retrieved documents."""

        score: int = Field(
            description="1: completely irrelevant; 2: marginally relevant; 3: moderately relevant; 4: highly relevant; 5: completely relevant."
        )
        
    prompt_wo_ctx_str = LongDocPrompt.remove_multi_ws("""
        User question:
        {question}
        
        Retrieved document:
        {document}
        
        You are a grader assessing relevance of the retrieved document to the user question.
        Give a score from 1 to 5 to indicate the relevance of the document from completely irrelevant to completely relevant.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        {format_instructions}
    """)
    
    prompt_w_ctx_str = LongDocPrompt.remove_multi_ws("""
        User question:
        {question}
        
        Useful documents:
        {context}
        
        New document:
        {document}
        
        You are a grader assessing relevance of a newly retrieved document to the user question.
        Some known useful documents are also provided.
        Give a score from 1 to 5 to indicate the relevance of the document from completely irrelevant to completely relevant.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        {format_instructions}
    """)
    
    def __init__(self, llm: BaseChatModel) -> None:
        output_parser = PydanticOutputParser(pydantic_object=self.GradeDocumentOutput)
        output_fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm, max_retries=2)
        prompt_wo_ctx = PromptTemplate(
            template=self.prompt_wo_ctx_str,
            input_variables=['document', 'question'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        prompt_w_ctx = PromptTemplate(
            template=self.prompt_w_ctx_str,
            input_variables=['document', 'question', 'context'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        self.chain_wo_ctx = prompt_wo_ctx | llm | output_fixing_parser
        self.chain_w_ctx = prompt_w_ctx | llm | output_fixing_parser
        
    def __call__(self, documents:List[str], questions:List[str], contexts:List[str]=[]) -> List[GradeDocumentOutput]:
        if contexts:
            return self.chain_w_ctx.batch([{'question': question, 'document': doc, 'context': ctx} for doc, question, ctx in zip(documents, questions, contexts)])
        else:
            return self.chain_wo_ctx.batch([{'question': question, 'document': doc} for doc, question in zip(documents, questions)])

# class AnalyzeDocument:
    
#     prompt_wo_ctx_str = LongDocPrompt.remove_multi_ws("""
#         You are an information extractor.
#         Extract the information in the given documents that is useful for answer the user question.
#         Only gather the useful information and DO NOT answer the question now.
        
#         Documents:
#         {document}
        
#         User question:
#         {question}
        
#         Useful Information:
#     """)
    
#     prompt_w_ctx_str = LongDocPrompt.remove_multi_ws("""
#         You are an information extractor.
#         Extract the information in the given documents that is useful for answer the user question.
#         Some currently known information is also provided.
#         You can make use of the known information to help you connect the document with the question.
#         Only gather the useful information and DO NOT answer the question now.
        
#         Known information:
#         {context}
        
#         Documents:
#         {document}
        
#         User question:
#         {question}
        
#         Useful Information:
#     """)
    
#     def __init__(self, llm: BaseChatModel) -> None:
#         output_parser = StrOutputParser()
#         prompt_wo_ctx = PromptTemplate(
#             template=self.prompt_wo_ctx_str,
#             input_variables=['document', 'question'],
#         )
#         prompt_w_ctx = PromptTemplate(
#             template=self.prompt_w_ctx_str,
#             input_variables=['document', 'question', 'context'],
#         )
#         self.chain_wo_ctx = prompt_wo_ctx | llm | output_parser
#         self.chain_w_ctx = prompt_w_ctx | llm | output_parser
        
#     def __call__(self, documents:List[str], questions:List[str], contexts:List[str]=[]):
#         if contexts:
#             return self.chain_w_ctx.batch([{'question': question, 'document': doc, 'context': ctx} for doc, question, ctx in zip(documents, questions, contexts)])
#         else:
#             return self.chain_wo_ctx.batch([{'question': question, 'document': doc} for doc, question in zip(documents, questions)])
    
        
class GenerateAnswer:
    
    prompt:ChatPromptTemplate = hub.pull("rlm/rag-prompt")
    
    def __init__(self, llm: BaseChatModel) -> None:
        self.chain = self.prompt | llm | StrOutputParser()
        
    def __call__(self, contexts:List[str], questions:List[str]):
        return self.chain.batch([{'question': question, 'context': context} for context, question in zip(contexts, questions)])
        
        
class EvalCompleteInfo:
    
    class EvalCompleteInfoOutput(BaseModel):
        """Binary score for complete information check on context."""

        binary_score: Literal['yes', 'no'] = Field(
            description="The context contains complete information to answer the question, yes or no"
        )
        
    prompt_str = LongDocPrompt.remove_multi_ws("""
        You are a grader assessing the completeness of a context to answer a user question.
        If the context contains enough information to support a confident answer for the question, grade it as complete.
        It need to be a stringent evaluation, as the next step is to generate the final answer for the question. The goal is to avoid any missing information.
        Give a binary score 'yes' or 'no' score to indicate whether the context is complete to the question.
        DO NOT answer the question now.
        {format_instructions}
        
        Context:
        {context}
        
        User question:
        {question}
    """)
    
    def __init__(self, llm: BaseChatModel) -> None:
        output_parser = PydanticOutputParser(pydantic_object=self.EvalCompleteInfoOutput)
        output_fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm, max_retries=2)
        prompt = PromptTemplate(
            template=self.prompt_str,
            input_variables=['context', 'question'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        self.chain = prompt | llm | output_fixing_parser
        
    def __call__(self, contexts:List[str], questions:List[str]) -> List[EvalCompleteInfoOutput]:
        return self.chain.batch([{'question': question, 'context': context} for context, question in zip(contexts, questions)])


class GenerateQuery:

    prompt_str = LongDocPrompt.remove_multi_ws("""
        Write queries in natural language to retrieve information relevant to a user question.
        You may generate a single query or multiple queries, depending on the information you need.
        Generate the queries directly and DO NOT explain your answer.
        
        User question:
        {question}
        
        Queries:
    """)
    
    prompt_w_doc_str = LongDocPrompt.remove_multi_ws("""
        Write queries in natural language to retrieve information relevant to a user question.
        Some known useful documents are also provided.
        Generate the queries directly and DO NOT explain your answer.
        
        Useful documents:
        {document}
        
        User question:
        {question}
        
        Queries:
    """)
    
    def __init__(self, llm:BaseChatModel) -> None:
        output_parser = StrOutputParser()
        
        prompt = PromptTemplate(
            template=self.prompt_str,
            input_variables=['question'],
        )
        
        prompt_w_doc = PromptTemplate(
            template=self.prompt_w_doc_str,
            input_variables=['document', 'question'],
        )
        
        self.chain = prompt | llm | output_parser
        self.chain_w_doc = prompt_w_doc | llm | output_parser
        
    def __call__(self, questions:List[str], documents:List[str]=[]):
        has_document = len(documents) > 0
        if has_document:
            return self.chain_w_doc.batch([{'question': question, 'document': document} for question, document in zip(questions, documents)])
        else:
            return self.chain.batch([{'question': question} for question in questions])


class TreeQuery:
    
    class TreeQueryNextChunkOutput(BaseModel):
        """The subsummary to explore."""

        explore_chunk: int = Field(
            description="The id of the subsummary to explore."
        )
    
    prompt_chunk_str = LongDocPrompt.remove_multi_ws("""
        Useful documents:
        {document}
        
        Summaries:
        {summary}
        
        User question:
        {question}
        
        You are given a user question, several relevant documents and a list of summaries.
        Each summary is a concatenation of subsummaries in sequential order, and each subsummary is the summary of a lower-level summary or document.
        To retrieve more documents useful to the user question, determine a subsummary that you would like to explore.
        Choosing a higher level subsummary will result in a broader range of document search, while choosing a lower level subsummary will result in a more precise document search.
        Make the decision based on the kind of information you need.
        Please DO NOT choose subsummaries from {known_chunk}. These documents of these subsummaries are already explored.
        
        {format_instructions}
    """)
    
    prompt_query_str = LongDocPrompt.remove_multi_ws("""
        You are given a user question, several relevant documents and a list of summaries.
        Each summary is a concatenation of subsummaries in sequential order, and each subsummary is the summary of a lower-level summary or document.
        We are planning to explore more useful information under summary {chunk}.
        Write queries in natural language to retrieve useful documents under summary {chunk}.
        You should make use of the subsummary of summary {chunk} below to help you infer the possibly useful under summary {chunk}.
        Generate the queries directly and DO NOT explain your answer.
        
        Useful documents:
        {document}
        
        Summaries:
        {summary}
        
        User question:
        {question}
        
        Queries:
    """)
    
    def __init__(self, llm:BaseChatModel) -> None:
        output_parser = PydanticOutputParser(pydantic_object=self.TreeQueryNextChunkOutput)
        output_fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm, max_retries=3)
        
        query_output_parser = StrOutputParser()
        
        prompt_chunk = PromptTemplate(
            template=self.prompt_chunk_str,
            input_variables=['document', 'summary', 'question', 'known_chunk'],
            partial_variables={'format_instructions': output_parser.get_format_instructions()}
        )
        prompt_query = PromptTemplate(
            template=self.prompt_query_str,
            input_variables=['document', 'chunk', 'summary', 'question'],
        )
        self.chain_chunk = prompt_chunk | llm | output_fixing_parser
        self.chain_query = prompt_query | llm | query_output_parser
        
    def __call__(self, questions:List[str], documents:List[str], summaries:List[str], known_chunks:List[List[int]]) -> List[TreeQueryNextChunkOutput]:
        return self.chain_chunk.batch([{'question': question, 'document': document, 'summary': summary, 'known_chunk': known_chunk} for question, document, summary, known_chunk in zip(questions, documents, summaries, known_chunks)])
        
    def get_query(self, questions:List[str], contexts:List[str], summaries:List[str], chunks:List[int]):
        return self.chain_query.batch([{'question': question, 'document': context, 'chunk': chunk, 'summary': summary} for context, question, chunk, summary in zip(contexts, questions, chunks, summaries)])
        
class NavigateState(BaseModel):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    dpr_queries: List[str] = []
    branch_queries: List[List[str]] = []
    branch_roots: List[int] = []
    retriever: Literal['dpr', 'tree'] = 'dpr'
    document_ids: List[int] = []
    new_document_ids: List[int] = []
    retrieve_cnts: int = 0
    answer:str = ''
    answer_file:str

class NavigateAgent:
    class Nodes:
        REFORM_QUERY = 'reform_query'
        RETRIEVE_DOC = 'retrieve_doc'
        GRADE_DOC = 'grade_doc'
        ANALYZE_DOC = 'analyze_doc'
        GENERATE_ANSWER = 'generate_answer'
    
    def __init__(self, llm:BaseChatModel, max_retrieve_turn:int=5) -> None:
        self.llm = llm
        self.dpr_retriever: MyDPR
        self.tree_retriever: MyTree
        self.documents: List[Document]
        self.reform_query_chain = GenerateQuery(self.llm)
        self.tree_query_chain = TreeQuery(self.llm)
        self.grade_doc_chain = GradeDocument(self.llm)
        self.eval_complete_info_chain = EvalCompleteInfo(self.llm)
        self.generate_answer_chain = GenerateAnswer(self.llm)
        self.max_retrieve_turn = max_retrieve_turn
        
    def current_info(self, state:NavigateState):
        state.document_ids.sort()
        current_documents = '\n\n'.join([f'Document {doc_id}: {self.documents[doc_id].page_content}' for doc_id in state.document_ids])
        return current_documents
    
    def parse_summary_info(self, summary:str):
        chunk_summaries = self.splitlines(summary)
        new_chunk_summaries = []
        for chunk_sum in chunk_summaries:
            if chunk_sum.startswith('Chunk'):
                chunk_id, chunk_info = chunk_sum.split(':', 1)
                chunk_id = int(chunk_id.strip().split()[1])
                chunk_info = chunk_info.strip()
                if chunk_id in self.tree_retriever[0]:
                    chunk_sum = f'Subsummary of Document {chunk_id}: {chunk_info}'
                else:
                    chunk_sum = f'Subsummary of Summary {chunk_id}: {chunk_info}'
                new_chunk_summaries.append(chunk_sum)
        chunk_summaries_str = '\n'.join(new_chunk_summaries)
        return chunk_summaries_str
    
    def splitlines(self, text:str):
        return [t.strip() for t in text.splitlines() if t.strip()]
    
    def reform_query(self, state:NavigateState):
        current_documents = self.current_info(state)
        state.branch_queries.clear()
        state.branch_roots.clear()
        state.dpr_queries.clear()
        if state.document_ids and state.retriever == 'tree':
            summaries:List[str] = []
            branches:List[List[int]] = []
            for doc_id in state.document_ids:
                for branch in self.tree_retriever.retrieve_branches(self.tree_retriever.docs[doc_id]):
                    branch = tuple(branch[1:])
                    if branch not in branches:
                        branches.append(branch)
                        summaries.append('\n\n'.join([
                            f'Summary {node_idx}:\n{self.parse_summary_info(self.tree_retriever.docs[node_idx].index.summary)}' 
                            for node_idx in branch
                        ]))
            
            batch_size = len(summaries)
            batch_results = self.tree_query_chain(
                questions=[state.question] * batch_size, 
                documents=[current_documents] * batch_size, 
                summaries=summaries, 
                known_chunks=[state.document_ids] * batch_size
            )
            
            root_summary_pairs = [(result.explore_chunk, summary) for result, summary in zip(batch_results, summaries) if any([result.explore_chunk in level for level in self.tree_retriever.tree])]
            
            if root_summary_pairs:
                batch_size = len(root_summary_pairs)
                roots, summaries = zip(*root_summary_pairs)
                batch_queries = self.tree_query_chain.get_query(
                    questions=[state.question] * batch_size, 
                    contexts=[current_documents] * batch_size, 
                    summaries=summaries, 
                    chunks=roots)
                root2queries = defaultdict(list)
                for root, queries in zip(roots, batch_queries):
                    root2queries[root].extend(self.splitlines(queries))
                state.branch_roots, state.branch_queries = zip(*root2queries.items())
        else:
            documents = [current_documents] if state.document_ids else []
            questions = [state.question]
            result = self.reform_query_chain(questions=questions, documents=documents)[0]
            state.dpr_queries = self.splitlines(result)
        state.new_document_ids = list(set(state.new_document_ids))
        return state
    
    def retrieve_doc(self, state:NavigateState):
        state.new_document_ids.clear()
        queries:List[str] = []
        retrieve_ranges:List[Set[int]] = []
        chunk2scores:Dict[int, List[float]] = defaultdict(list)
        if state.branch_roots:
            for branch_root, branch_query in zip(state.branch_roots, state.branch_queries):
                children = self.tree_retriever.retrieve_children(self.tree_retriever.docs[branch_root])
                queries.extend(branch_query)
                retrieve_ranges.extend([set(children)] * len(branch_query))
        else:
            queries = state.dpr_queries if state.dpr_queries else [state.question]
            retrieve_ranges = [set(range(len(self.documents)))] * len(queries)
            
        assert len(queries) == len(retrieve_ranges)
        
        for query, retrieve_range in zip(queries, retrieve_ranges):
            for doc, score in self.dpr_retriever.vectorstore.similarity_search_with_score(query, k=len(self.documents)):
                if doc.metadata['i'] in retrieve_range:
                    chunk2scores[doc.metadata['i']].append(score)
                    
        chunk2score = [(chunk, np.mean(scores)) for chunk, scores in chunk2scores.items()]
        chunk2score.sort(key=lambda x: x[1])
        state.new_document_ids = [chunk for chunk, _ in chunk2score if chunk not in state.document_ids][:4]
        state.retrieve_cnts += 1
        return state
    
    def grade_doc(self, state:NavigateState):
        current_documents = self.current_info(state)
        documents = [self.documents[doc_id].page_content for doc_id in state.new_document_ids]
        batch_size = len(state.new_document_ids)
        questions = [state.question] * batch_size
        contexts = ([current_documents] * batch_size) if state.document_ids else []
        
        batch_results = self.grade_doc_chain(documents=documents, questions=questions, contexts=contexts)
        state.new_document_ids = [doc_id for doc_id, result in zip(state.new_document_ids, batch_results) if result.score > 1]
        return state
    
    def check_non_empty_retrieval(self, state:NavigateState):
        if state.new_document_ids:
            return 'Not empty'
        elif state.retrieve_cnts < self.max_retrieve_turn:
            return 'Empty'
        else:
            return 'Reach max retrieval'
    
    def analyze_doc(self, state:NavigateState):
        state.document_ids.extend(state.new_document_ids)
        return state
    
    def eval_complete_info(self, state:NavigateState):
        current_documents = self.current_info(state)
        result = self.eval_complete_info_chain(contexts=[current_documents], questions=[state.question])[0]
        if 'yes' in result.binary_score.lower() or state.retrieve_cnts >= self.max_retrieve_turn or len(state.document_ids) >= 6:
            return 'generate_answer'
        else:
            return 'update_query'
    
    def generate_answer(self, state:NavigateState):
        current_documents = self.current_info(state)
        state.answer = self.generate_answer_chain(contexts=[current_documents], questions=[state.question])[0]
        write_json(state.answer_file, state.dict())
        return state
    
    def create_workflow(self, dpr_retriever:MyDPR, tree_retriever:MyTree, documents:List[Document]):
        self.dpr_retriever = dpr_retriever
        self.tree_retriever = tree_retriever
        self.documents = documents
        workflow = StateGraph(NavigateState)
        for attr_name, attr_value in vars(self.Nodes).items():
            if not attr_name.startswith('_'):
                workflow.add_node(attr_value, getattr(self, attr_value))
        
        workflow.set_entry_point(self.Nodes.RETRIEVE_DOC)
        workflow.add_edge(self.Nodes.RETRIEVE_DOC, self.Nodes.GRADE_DOC)
        workflow.add_conditional_edges(
            self.Nodes.GRADE_DOC,
            self.check_non_empty_retrieval,
            {
                'Not empty': self.Nodes.ANALYZE_DOC,
                'Empty': self.Nodes.REFORM_QUERY,
                'Reach max retrieval': self.Nodes.GENERATE_ANSWER
            },
        )
        workflow.add_conditional_edges(
            self.Nodes.ANALYZE_DOC,
            self.eval_complete_info,
            {
                "update_query": self.Nodes.REFORM_QUERY,
                "generate_answer": self.Nodes.GENERATE_ANSWER,
            },
        )
        workflow.add_edge(self.Nodes.REFORM_QUERY, self.Nodes.RETRIEVE_DOC)
        workflow.add_edge(self.Nodes.GENERATE_ANSWER, END)
        app = workflow.compile()
        
        return app
    
