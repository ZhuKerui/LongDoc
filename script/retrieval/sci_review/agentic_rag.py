from .pipeline_base import *

from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.prebuilt.chat_agent_executor import AgentState


class AgenticRAG(MyPipeline):
    
    TOOLS = "tools"
    TOOL_POST_PROCESS = "tool_post_process"
    
    PASSAGES = "passages"
    QUESTION = "question"
    RETRIEVAL = "retrieval"
    CHUNKS = "chunks"
    
    class AgenticRAGState(AgentState):
        passages: list[int]
        question: str
        retrieval: list[int]
        chunks: list[str]
    
    def __init__(self, enable_trace = False, project_name = None, llm_model = GPT_MODEL_CHEAP):
        super().__init__(enable_trace, project_name, llm_model)
        
        
    def load_langgraph(self, tools:list[BaseTool]):
        assert tools, "No tools provided"
        
        self.tools = tools
        
        # Build Workflow
        workflow = StateGraph(self.AgenticRAGState)

        # Define the nodes
        workflow.add_node(self.AGENT, self.agent)
        workflow.add_edge(START, self.AGENT)
        
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

        # Compile
        self.app = workflow.compile()
        
        
    def invoke(self, question:str, chunks:list[str]):

        # Run
        inputs = {
            self.MESSAGES: [("user", question)],
            self.PASSAGES: [],
            self.QUESTION: question,
            self.RETRIEVAL: [],
            self.CHUNKS: chunks
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
        messages:list[BaseMessage] = state[self.MESSAGES]
        
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        # We return a list, because this will get added to the existing list
        return {self.MESSAGES: [response]}


    def tool_post_process(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        passages:list[int] = state[self.PASSAGES]
        question:str = state[self.QUESTION]
        chunks:list[str] = state[self.CHUNKS]
        retrieval = list[int]()
        retrieval_called = False
        
        tool_messages = list[ToolMessage]()
        ai_message:AIMessage = None
        
        for message in state[self.MESSAGES][::-1]:
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
                temp_retrieval = [chunks.index(chunk) for chunk in tool_message.content.split(CHUNK_SEP)]
                retrieval.extend(temp_retrieval)
                new_doc_ids = [doc_id for doc_id in temp_retrieval if doc_id not in passages]
                
                if new_doc_ids:
                    
                    # Chain
                    chain = grade_doc_prompt | self.llm.with_structured_output(grade_doc_data)
                
                    scored_results:list[grade_doc_data] = chain.batch([{"question": question, "context": chunks[doc_id]} for doc_id in new_doc_ids])

                    new_relevant_doc_ids = [doc_id for doc_id, scored_result in zip(new_doc_ids, scored_results) if scored_result.binary_score == "yes"]
                
                    passages.extend(new_relevant_doc_ids)

                    if new_relevant_doc_ids:
                        print("---NEW RELEVANT DOCS---")
                        tool_message.content = 'New relevant documents are retrieved.\n\n' + PARAGRAPH_SEP.join(chunks[doc_id] for doc_id in new_relevant_doc_ids)

                    else:
                        print("---DECISION: DOCS NOT RELEVANT---")
                        tool_message.content = "New documents are retrieved but none are relevant to the user question."
                else:
                    print("---DUPLICATED RETRIEVAL---")
                    tool_message.content = "Retrieved documents are already in previous steps. No new documents retrieved."
        
        if retrieval_called:
            return {self.PASSAGES: passages, self.RETRIEVAL: retrieval}
        
        
    @staticmethod
    def dump_process(process:list[dict], dump_file:str):
        dir = os.path.dirname(dump_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for a in process:
            for k, v in a.items():
                if k in {MyPipeline.AGENT, AgenticRAG.TOOLS}:
                    for mid in range(len(v[MyPipeline.MESSAGES])):
                        v[MyPipeline.MESSAGES][mid] = v[MyPipeline.MESSAGES][mid].model_dump()
        with open(dump_file, 'w') as f:
            json.dump(process, f)
            
            
    @staticmethod
    def load_process(dump_file:str) -> list[dict]:
        with open(dump_file) as f:
            process = json.load(f)
        for a in process:
            for k, v in a.items():
                if k == MyPipeline.AGENT:
                    v[MyPipeline.MESSAGES][0] = AIMessage(**(v[MyPipeline.MESSAGES][0]))
                elif k == AgenticRAG.TOOLS:
                    v[MyPipeline.MESSAGES][0] = ToolMessage(**(v[MyPipeline.MESSAGES][0]))
        return process
    
    @staticmethod
    def get_chunk_ids_from_process(
        process:list[dict],
        relevant_only:bool=True
    ) -> list[int]:
        if relevant_only:
            for step in process[::-1]:
                if AgenticRAG.TOOL_POST_PROCESS in step and step[AgenticRAG.TOOL_POST_PROCESS]:
                    return step[AgenticRAG.TOOL_POST_PROCESS][AgenticRAG.PASSAGES]
        else:
            chunk_ids = list[int]()
            for step in process:
                if AgenticRAG.TOOL_POST_PROCESS in step and step[AgenticRAG.TOOL_POST_PROCESS]:
                    chunk_ids.extend(step[AgenticRAG.TOOL_POST_PROCESS][AgenticRAG.RETRIEVAL])
            return list(set(chunk_ids))
            