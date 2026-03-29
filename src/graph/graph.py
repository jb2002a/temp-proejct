from langgraph.graph import StateGraph, START, END
from src.graph.state import State
from src.graph.nodes.retriever_node import retriever_node
from src.graph.nodes.generate_node import generate_node

def build_graph() -> StateGraph:
    """
    Build the graph
    """
    builder = StateGraph(State)
    builder.add_node(retriever_node)
    builder.add_node(generate_node)
    builder.add_edge(START, "retriever_node") 
    builder.add_edge("retriever_node", "generate_node")
    builder.add_edge("generate_node", END)
    graph = builder.compile()
    return graph

if __name__ == "__main__":
    # python -m src.graph
    query = input("Enter your query: ")
    graph = build_graph()
    result = graph.invoke({"query": query})
    result = result["answer"]
    print(result)
