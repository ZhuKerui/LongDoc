from .pipeline_base import *

from langgraph.prebuilt.chat_agent_executor import AgentState
from nltk import sent_tokenize


class CitationGeneration(MyPipeline):
    
    PASSAGES = "passages"
    EXTRACTIONS = "extractions"
    
    class CitationGenerationState(AgentState):
        passages: list[int]
        extractions: list[tuple[str, list[int]]]
    
    def __init__(self, enable_trace = False, project_name = None, llm_model = GPT_MODEL_CHEAP):
        super().__init__(enable_trace, project_name, llm_model)
        
        
    def load_langgraph(self):
        
        # Build Workflow
        workflow = StateGraph(self.CitationGenerationState)

        # Define the nodes
        workflow.add_node(self.AGENT, self.agent)
        workflow.add_edge(START, self.AGENT)
        workflow.add_edge(self.AGENT, END)

        # Compile
        self.app = workflow.compile()
        
        
    def invoke(self, question:str, chunks:list[str]):

        # Run
        content = PARAGRAPH_SEP.join([f'Chunk {cid}: {chunk}' for cid, chunk in enumerate(chunks)])
        
        citation_generation_prompt = f'Below are text chunks from a paper:\n\n\n\n{content}\n\n\n\nBased on the provided text chunks, answer the following question:\n\n{question}\n\nWrite a detailed and well-structured paragraph in response to the question. Each statement in your answer must be explicitly supported by the provided text chunks. Cite the relevant chunks in parentheses after each supporting statement.\nExample format:\n"This work uses the BERT model as its base model (Chunk 1). The model is trained on the SQuAD dataset (Chunk 2, 3)."'
        
        inputs = {
            self.MESSAGES: [("user", citation_generation_prompt)],
            self.PASSAGES: []
        }
        
        return list(self.app.stream(inputs, output_keys=[self.MESSAGES, self.PASSAGES, self.EXTRACTIONS]))

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
        citation:str
        for citation in re.findall(r'\(Chunk \d+(?:, \d+)*\)', response.content):
            passages.extend(map(int, citation[7:-1].split(', ')))
            
        passages = list(set(passages))
        extraction_texts = sent_tokenize(response.content)
        extractions = [(text, [int(num_str) for citation in re.findall(r'\(Chunk \d+(?:, \d+)*\)', text) for num_str in citation[7:-1].split(', ')]) for text in extraction_texts]
                    
        # We return a list, because this will get added to the existing list
        return {self.MESSAGES: [response], self.PASSAGES: passages, self.EXTRACTIONS: extractions}

    @staticmethod
    def dump_process(process:list[dict], dump_file:str):
        dir = os.path.dirname(dump_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for a in process:
            for k, v in a.items():
                if k in {CitationGeneration.AGENT}:
                    for mid in range(len(v[CitationGeneration.MESSAGES])):
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
        return process[0][MyPipeline.AGENT][CitationGeneration.PASSAGES]
    
    @staticmethod
    def get_extraction_from_process(
        process:list[dict],
    ) -> list[str]:
        return process[0][MyPipeline.AGENT][CitationGeneration.EXTRACTIONS]