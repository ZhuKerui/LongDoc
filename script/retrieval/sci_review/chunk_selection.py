from .pipeline_base import *

from langgraph.prebuilt.chat_agent_executor import AgentState


class ChunkSelection(MyPipeline):
    
    PASSAGES = "passages"
    
    class ChunkSelectionType(enum.Enum):
        SELECTION = "selection"
        CLASSIFICATION = "classification"
    
    class ChunkSelectionState(AgentState):
        passages: list[int]
    
    def __init__(self, enable_trace = False, project_name = None, llm_model = GPT_MODEL_CHEAP):
        super().__init__(enable_trace, project_name, llm_model)
        
        
    def load_langgraph(self, selection_type:ChunkSelectionType):
        
        self.selection_type = selection_type
        
        # Build Workflow
        workflow = StateGraph(self.ChunkSelectionState)

        # Define the nodes
        workflow.add_node(self.AGENT, self.agent)
        workflow.add_edge(START, self.AGENT)
        workflow.add_edge(self.AGENT, END)

        # Compile
        self.app = workflow.compile()
        
        
    def invoke(self, question:str, chunks:list[str]):

        # Run
        content = PARAGRAPH_SEP.join([f'Chunk {cid}: {chunk}' for cid, chunk in enumerate(chunks)])
        
        match self.selection_type:
            case self.ChunkSelectionType.SELECTION:
                chunk_selection_prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nSelect the Chunk ids that are relevant to the following question: \n\n{question}\n\nReturn only the selected chunk ids separated by commas, e.g. "1, 3, 5".'
            case self.ChunkSelectionType.CLASSIFICATION:
                chunk_selection_prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nFor each chunk, decide whether it is relevant to the following question: \n\n{question}\n\nReturn your decision in the following format:\n\n0: Yes\n1: No\n2: No\n...'
        inputs = {
            self.MESSAGES: [("user", chunk_selection_prompt)],
            self.PASSAGES: []
        }
        
        return list(self.app.stream(inputs, output_keys=[self.MESSAGES, self.PASSAGES]))

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
        response = self.llm.invoke(messages)
        passages = list[int]()
        
        match self.selection_type:
            case self.ChunkSelectionType.SELECTION:
                passages = list(map(int, response.content.split(', ')))
            case self.ChunkSelectionType.CLASSIFICATION:
                for line in response.content.split('\n'):
                    classification_text = re.search(r'\d+: \w+$', line)
                    if classification_text and 'yes' in classification_text.string.lower():
                        passages.append(int(classification_text.string.split(':')[0]))
                    
        # We return a list, because this will get added to the existing list
        return {self.MESSAGES: [response], self.PASSAGES: passages}

    @staticmethod
    def dump_process(process:list[dict], dump_file:str):
        dir = os.path.dirname(dump_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for a in process:
            for k, v in a.items():
                if k in {MyPipeline.AGENT}:
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
        return process
    
    @staticmethod
    def get_chunk_ids_from_process(
        process:list[dict],
    ) -> list[int]:
        return process[0][MyPipeline.AGENT][ChunkSelection.PASSAGES]