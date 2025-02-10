from .pipeline_base import *

from typing import List, Literal

from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState


class AgenticRAG(MyPipeline):
    
    AGENT = "agent"
    TOOLS = "tools"
    TOOL_POST_PROCESS = "tool_post_process"
    
    MESSAGES = "messages"
    PASSAGES = "passages"
    QUESTION = "question"
    RETRIEVAL = "retrieval"
    
    class AgenticRAGState(AgentState):
        passages: List[str]
        question: str
        retrieval: List[str]
    
    def __init__(self, enable_trace = False, project_name = None):
        super().__init__(enable_trace, project_name)
        
        
    def load_langgraph(self, tools:List[BaseTool] = []):
        
        self.tools = tools
        
        # Build Workflow
        workflow = StateGraph(self.AgenticRAGState)

        # Define the nodes
        workflow.add_node(self.AGENT, self.agent)
        workflow.add_edge(START, self.AGENT)
        
        if self.tools:
            workflow.add_node(self.TOOLS, ToolNode(self.tools))
            workflow.add_node(self.TOOL_POST_PROCESS, self.tool_post_process)

            # Decide whether to retrieve
            workflow.add_conditional_edges(
                self.AGENT,
                # Assess agent decision
                tools_condition,
                {
                    # Translate the condition outputs to nodes in our graph
                    "tools": self.TOOLS,
                    END: END,
                },
            )

            workflow.add_edge(self.TOOLS, self.TOOL_POST_PROCESS)
            workflow.add_edge(self.TOOL_POST_PROCESS, self.AGENT)
        else:
            workflow.add_edge(self.AGENT, END)

        # Compile
        self.app = workflow.compile()
        
        
    def invoke(self, question:str):

        # Run
        inputs = {
            self.MESSAGES: [("user", question)],
            self.PASSAGES: [],
            self.QUESTION: question,
            self.RETRIEVAL: []
        }
        
        return list(self.app.stream(inputs, output_keys=[self.MESSAGES, self.PASSAGES, self.RETRIEVAL]))

    ### Nodes
    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        print("---CALL AGENT---")
        messages:List[BaseMessage] = state["messages"]
        
        if self.tools:
            model = self.llm.bind_tools(self.tools)
        else:
            model = self.llm
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    def tool_post_process(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        passages:List[str] = state["passages"]
        question:str = state["question"]
        new_passages:List[str] = []
        new_retrieval:List[str] = []
        retrieval_called = False
        
        tool_messages = list[ToolMessage]()
        ai_message:AIMessage = None
        
        for message in state["messages"][::-1]:
            if isinstance(message, ToolMessage):
                tool_messages.insert(0, message)
            elif isinstance(message, AIMessage):
                ai_message = message
                break
            else:
                raise ValueError("Invalid message type")
            
        assert len(ai_message.tool_calls) == len(tool_messages)
        for tool_message, tool_call in zip(tool_messages, ai_message.tool_calls):
            
            if tool_message.name.startswith("Retrieve"):
                retrieval_called = True
                print("---CHECK RELEVANCE---")
                retrieval = tool_message.content.split(CHUNK_SEP)
                new_retrieval.extend(f"{tool_call['args']['query']}{CHUNK_SEP}{doc}" for doc in retrieval)
                new_docs = [doc for doc in retrieval if doc not in passages]
                
                if new_docs:
                    # Sanity check
                    chunk_set = {chunk.page_content for chunk in self.doc_manager.chunks}
                    assert all(doc in chunk_set for doc in new_docs)
                    
                    # Chain
                    chain = grade_doc_prompt | self.llm.with_structured_output(grade_doc_data)
                
                    # scored_results:List[grade_doc_data] = chain.batch([{"question": tool_call['args']['query'], "context": doc} for doc in new_docs])
                    scored_results:List[grade_doc_data] = chain.batch([{"question": question, "context": doc} for doc in new_docs])

                    new_relevant_docs = [doc for doc, scored_result in zip(new_docs, scored_results) if scored_result.binary_score == "yes"]
                
                    new_passages.extend(new_relevant_docs)

                    if new_relevant_docs:
                        print("---NEW RELEVANT DOCS---")
                        tool_message.content = 'New relevant documents are retrieved.\n\n' + PARAGRAPH_SEP.join(new_relevant_docs)

                    else:
                        print("---DECISION: DOCS NOT RELEVANT---")
                        tool_message.content = "New documents are retrieved but none are relevant to the user question."
                else:
                    print("---DUPLICATED RETRIEVAL---")
                    tool_message.content = "Retrieved documents are already in previous steps. No new documents retrieved."
        
        if retrieval_called:
            passages.extend(new_passages)
            return {self.PASSAGES: passages, self.RETRIEVAL: new_retrieval}
        
        
    @staticmethod
    def dump_process(process:list[dict], dump_file:str):
        dir = os.path.dirname(dump_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for a in process:
            for k, v in a.items():
                if k in {AgenticRAG.AGENT, AgenticRAG.TOOLS}:
                    for mid in range(len(v['messages'])):
                        v['messages'][mid] = v['messages'][mid].model_dump()
        with open(dump_file, 'w') as f:
            json.dump(process, f)
            
            
    @staticmethod
    def load_process(dump_file:str) -> list[dict]:
        with open(dump_file) as f:
            process = json.load(f)
        for a in process:
            for k, v in a.items():
                if k == AgenticRAG.AGENT:
                    v['messages'][0] = AIMessage(**(v['messages'][0]))
                elif k == AgenticRAG.TOOLS:
                    v['messages'][0] = ToolMessage(**(v['messages'][0]))
        return process