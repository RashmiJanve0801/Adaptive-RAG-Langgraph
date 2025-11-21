from langgraph.graph import END, StateGraph, START
from src.States.state import GraphState
from functools import partial

def build_graph(components):
    """Build the LangGraph workflow"""
    from src.Nodes.node import (
        retrieve, generate, grade_documents, transform_query, web_search,
        route_question, decide_to_generate, grade_generation_v_documents_and_question
    )
    
    workflow = StateGraph(GraphState)
    
    # Create partial functions with components
    retrieve_node = partial(retrieve, components=components)
    generate_node = partial(generate, components=components)
    grade_documents_node = partial(grade_documents, components=components)
    transform_query_node = partial(transform_query, components=components)
    web_search_node = partial(web_search, components=components)
    route_question_edge = partial(route_question, components=components)
    decide_to_generate_edge = partial(decide_to_generate, components=components)
    grade_generation_edge = partial(grade_generation_v_documents_and_question, components=components)
    
    # Define the nodes
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("transform_query", transform_query_node)
    
    # Build graph
    workflow.add_conditional_edges(
        START,
        route_question_edge,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate_edge,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_edge,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )
    
    # Compile
    app = workflow.compile()
    return app
