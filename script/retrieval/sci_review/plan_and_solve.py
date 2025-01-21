from .base import *

from typing_extensions import TypedDict


# Data state
class GraphState(TypedDict):
    question: str
    full_generation: str
    answer: str


class PlanAndSolve(MyPipeline):
    def __init__(self, doc_file = None, doc_str = None, enable_trace = False, project_name = None):
        super().__init__(doc_file, doc_str, enable_trace, project_name)
    
    def update_doc(self, doc_file = None, doc_str = None):
        self.doc_file = doc_file
        self.doc_str = doc_str
        
        # Load Document
        if self.doc_file is not None:
            self.doc_str = PyMuPDFLoader(self.doc_file).load()
        elif self.doc_str is None:
            raise NotImplementedError
        
    
    def load_langgraph(self):
        # Load Document
        if self.doc_file is not None:
            self.doc_str = PyMuPDFLoader(self.doc_file).load()
        elif self.doc_str is None:
            raise NotImplementedError

        # Load Plan and Solve
        plan_and_solve_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You need to complete a task over a given paper.")
                ("human", MyPipeline.remove_tab("""
                    Paper: \n\n {paper} \n\n Task: {question}
                    
                    Let's first understand the task, extract critical concepts and relationships from the task description and devise a complete plan to resolve the concepts and relationships from the paper. Then, let's carry out the plan, extract the relevant information from the paper, solve the problem step by step, and show the answer.
                """)),
            ]
        )
        self.plan_and_solve_chain = plan_and_solve_prompt | self.llm | StrOutputParser()
        
        # Load Answer Extraction
        answer_extraction_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You need to extract the answer part from the given problem solving steps. Only extract the answer and do not generate any explanation.")
                ("human", "Task: \n\n {question} \n\n Problem solving steps: {steps}"),
            ]
        )
        self.answer_extraction_chain = answer_extraction_prompt | self.llm | StrOutputParser()
        
        
        # Build Workflow
        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("plan_and_solve", self.plan_and_solve)
        workflow.add_node("answer_extraction", self.answer_extraction)

        # Build graph
        workflow.add_edge(START, "plan_and_solve")
        workflow.add_edge("plan_and_solve", "answer_extraction")
        workflow.add_edge("answer_extraction", END)

        # Compile
        self.app = workflow.compile()
        
        
    def invoke(self, question:str):
        inputs = {"question": question}
        return self.app.stream(inputs)


    ### Nodes
    
    def plan_and_solve(self, state):
        question = state["question"]
        paper = self.doc_str

        # Retrieval
        full_generation = self.plan_and_solve_chain.invoke({"paper": paper, "question": question})
        return {'question': question, 'full_generation': full_generation}


    def answer_extraction(self, state):
        question = state["question"]
        full_generation = state["full_generation"]

        # RAG generation
        answer = self.answer_extraction_chain.invoke({"question": question, "steps": full_generation})
        return {'question': question, 'full_generation': full_generation, 'answer': answer}


