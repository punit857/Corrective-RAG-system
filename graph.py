from langgraph.graph import StateGraph, START, END
from state import GraphState


from nodes import (
    retrieve_node,
    eval_each_doc_node,
    rewrite_query_node,
    web_search_node,
    refine,
    generate
)

def route_after_eval(state: GraphState) -> str:
    verdict = state.get("verdict")
    
    if verdict == "CORRECT":
        print("---ROUTING: Data is perfect, skipping web search -> REFINE---")
        return "refine"
    else:
        print("---ROUTING: Missing data, triggering fallback -> REWRITE QUERY---")
        return "rewrite_query"



def build_crag_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("eval_each_doc", eval_each_doc_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("refine", refine)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "eval_each_doc")

    workflow.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",                 
            "rewrite_query": "rewrite_query",   
        }
    )

    
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "refine")

    workflow.add_edge("refine", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

crag_app = build_crag_graph()