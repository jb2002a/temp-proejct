from typing_extensions import TypedDict
from typing import List
from llama_index.core.schema import NodeWithScore

class State(TypedDict):
    query: str
    nodes: List[NodeWithScore]
    answer: str

    