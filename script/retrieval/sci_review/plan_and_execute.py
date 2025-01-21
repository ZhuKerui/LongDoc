from .base import *
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import VectorStoreQATool

from langchain import hub

from langgraph.prebuilt import create_react_agent
import operator
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
    

class PlanAndExecute(MyPipeline):
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

        tools = [VectorStoreQATool()]
        
        self.react_agent_chain = create_react_agent(self.llm, tools, state_modifier=hub.pull("ih/ih-react-agent-executor"))

        planner_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MyPipeline.remove_tab("""
                    For the given objective, come up with a simple step by step plan. This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.
                    """),
                ),
                ("placeholder", "{messages}"),
            ]
        )
        
        self.planner = planner_prompt | self.llm.with_structured_output(Plan)

        replanner_prompt = ChatPromptTemplate.from_template(
            MyPipeline.remove_tab("""
                For the given objective, come up with a simple step by step plan. This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

                Your objective was this:
                {input}

                Your original plan was this:
                {plan}

                You have currently done the follow steps:
                {past_steps}

                Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
        ))

        self.replanner = replanner_prompt | self.llm.with_structured_output(Act)

        workflow = StateGraph(PlanExecute)

        # Add the plan node
        workflow.add_node("planner", self.plan_step)

        # Add the execution step
        workflow.add_node("agent", self.execute_step)

        # Add a replan node
        workflow.add_node("replan", self.replan_step)

        workflow.add_edge(START, "planner")

        # From plan we go to agent
        workflow.add_edge("planner", "agent")

        # From agent, we replan
        workflow.add_edge("agent", "replan")

        workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            self.should_end,
            ["agent", END],
        )

        # Finally, we compile it!
        # This compiles it into a LangChain Runnable,
        # meaning you can use it as you would any other runnable
        self.app = workflow.compile()



    async def execute_step(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:\n{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        agent_response = await self.react_agent_chain.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }


    async def plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}


    async def replan_step(self, state: PlanExecute):
        output = await self.replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}


    def should_end(self, state: PlanExecute):
        if "response" in state and state["response"]:
            return END
        else:
            return "agent"


