from .base import *
from pprint import pprint
from langchain_community.vectorstores import Chroma

from typing import List

from typing_extensions import TypedDict


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
    
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
    
    
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# Data state
class GraphState(TypedDict):
    question: str
    answer: str
    documents: List[str]


class SelfRAG(MyPipeline):
    def __init__(self, doc_file = None, doc_str = None, enable_trace = False, project_name = None, ret_num: int = 5):
        self.ret_num = ret_num
        super().__init__(doc_file, doc_str, enable_trace, project_name)
        
    def update_doc(self, doc_file = None, doc_str = None):
        self.clear_up()
        self.doc_file = doc_file
        self.doc_str = doc_str
        
        if self.doc_file is not None:
            self.doc_splits = PyMuPDFLoader(self.doc_file).load_and_split(self.text_splitter)
        elif self.doc_str is not None:
            self.doc_splits = self.text_splitter.create_documents([self.doc_str])
        else:
            raise NotImplementedError

        # Add to vectorDB
        self.vectorstore = Chroma.from_documents(
            documents=self.doc_splits,
            embedding=self.embedding,
        )
        self.retriever = self.vectorstore.as_retriever()
    
    def load_langgraph(self):
        # Load Embeddings
        self.emb_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': False}
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.emb_model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        
        # Load text splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.embedding._client.tokenizer,
            chunk_size=self.embedding._client.get_max_seq_length(), 
            chunk_overlap=0
        )
        
        # Load Document
        if self.doc_file is not None:
            self.doc_splits = PyMuPDFLoader(self.doc_file).load_and_split(self.text_splitter)
        elif self.doc_str is not None:
            self.doc_splits = self.text_splitter.create_documents([self.doc_str])
        else:
            raise NotImplementedError

        # Add to vectorDB
        self.vectorstore = Chroma.from_documents(
            documents=self.doc_splits,
            embedding=self.embedding,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={'k': self.ret_num}
        )
        
        # Load RAG
        self.rag_chain = hub.pull("rlm/rag-prompt") | self.llm | StrOutputParser()

        # Load doc grader
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SelfRAG.remove_tab("""
                    You are a grader assessing relevance of a retrieved document to a user question. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                """)),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        retrieval_grader = self.llm.with_structured_output(GradeDocuments)
        self.retrieval_grader_chain = grade_prompt | retrieval_grader
        
        # Load Hallucination Grader
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SelfRAG.remove_tab("""
                    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
                """)),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        hallucination_grader = self.llm.with_structured_output(GradeHallucinations)
        self.hallucination_grader_chain = hallucination_prompt | hallucination_grader
        
        # Load Answer Grader
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SelfRAG.remove_tab("""
                    You are a grader assessing whether an answer addresses / resolves a question. Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
                """)),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        answer_grader = self.llm.with_structured_output(GradeAnswer)
        self.answer_grader_chain = answer_prompt | answer_grader
        
        ### Question Re-writer
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SelfRAG.remove_tab("""
                    You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
                """)),
                ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        self.question_rewriter_chain = re_write_prompt | self.llm | StrOutputParser()
        
        # Build Workflow
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve
        workflow.add_node("grade_documents", self.grade_documents)  # grade documents
        workflow.add_node("generate", self.generate)  # generatae
        workflow.add_node("transform_query", self.transform_query)  # transform_query

        # Build graph
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        self.app = workflow.compile()
        
        
    def invoke(self, question:str):
        # docs = self.retriever.get_relevant_documents(question)
        # doc_txt = docs[1].page_content
        # print(self.retrieval_grader_chain.invoke({"question": question, "document": doc_txt}))

        # generation = self.rag_chain.invoke({"context": docs, "question": question})
        # print(generation)
        
        # self.hallucination_grader_chain.invoke({"documents": docs, "generation": generation})
        
        # self.answer_grader_chain.invoke({"question": question, "generation": generation})

        # self.question_rewriter_chain.invoke({"question": question})

        # Run
        inputs = {"question": question}
        return self.app.stream(inputs)


    ### Nodes
    
    def retrieve(self, state):
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}


    def generate(self, state):
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        answer = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "answer": answer}


    def grade_documents(self, state):
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score:GradeDocuments = self.retrieval_grader_chain.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                filtered_docs.append(d)
            else:
                continue
        return {"documents": filtered_docs, "question": question}


    def transform_query(self, state):
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter_chain.invoke({"question": question})
        return {"documents": documents, "question": better_question}


    ### Edges

    def decide_to_generate(self, state):
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            return "generate"


    def grade_generation_v_documents_and_question(self, state):
        question = state["question"]
        documents = state["documents"]
        answer = state["answer"]

        score:GradeHallucinations = self.hallucination_grader_chain.invoke(
            {"documents": documents, "generation": answer}
        )
        grade = score.binary_score

        # Check hallucination
        if grade == "yes":
            # Check question-answering
            score:GradeAnswer = self.answer_grader_chain.invoke({"question": question, "generation": answer})
            grade = score.binary_score
            if grade == "yes":
                return "useful"
            else:
                return "not useful"
        else:
            return "not supported"
        
        
    def clear_up(self):
        self.vectorstore.delete_collection()
        del self.vectorstore


