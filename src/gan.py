# Graph Agent Network
from .base_utils import *


class NodeInfo(MyIndex):
    internal_facts:List[str]
    external_facts:Dict[Dict[int, str]]
    
class AgentNode:
    def __init__(self) -> None:
        self.text
class GraphAgentNetwork:
    