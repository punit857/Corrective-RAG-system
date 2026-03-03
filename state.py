from typing import List, TypedDict

class GraphState(TypedDict):

    question: str 
    documents: List[str]       # All raw documents retrieved from the vector database
    good_documents: List[str]  # Only the documents that score high enough
    verdict: str               # Will be "CORRECT", "INCORRECT", or "AMBIGUOUS"
    reason: str                # A short explanation of why the verdict was chosen
    web_query: str             # The optimized search query written by the LLM
    web_documents: List[str]   # Fresh information pulled from Tavily
    refined_context: str       # The final, context made good documents and web documents(if required)
    generation: str            # The final answer provided to the user
    selected_files: List[str]