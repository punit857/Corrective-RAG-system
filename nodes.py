import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from state import GraphState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- THRESHOLDS ---
UPPER_TH = 0.7
LOWER_TH = 0.3

print("--- LOADING AND INDEXING PDF DOCUMENTS ---")
os.makedirs("documents", exist_ok=True)
pdf_files = [f for f in os.listdir("documents") if f.endswith(".pdf")]
docs = []


for pdf in pdf_files:
    docs.extend(PyPDFLoader(f"documents/{pdf}").load())

if not docs:
    print(" No PDFs found. Initializing empty database.")
    docs = [Document(page_content="Empty Knowledge Base.", metadata={"source": "system"})]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)


for d in chunks:
    clean_text = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
    source_file = os.path.basename(d.metadata.get("source", "Unknown_Document"))
    d.page_content = f"[Source Document: {source_file}]\n{clean_text}"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("--- DATABASE CREATED ---")


# --- NODE 0: THE RETRIEVER ---

def retrieve_node(state: GraphState):
    print("---RETRIEVING DOCUMENTS FROM DATABASE---")
    question = state["question"]
    selected_files = state.get("selected_files") 
    
    
    if selected_files is not None and len(selected_files) == 0:
        print("User unselected all files. Skipping local database.")
        return {"documents": []}
        
    
    search_kwargs = {"k": 8, "fetch_k": 30}
    
    # If specific files are selected, apply the Metadata Filter
    if selected_files:
        print(f"Filtering database for: {selected_files}")
        
        
        # We use os.path.basename to strip "documents/" so it matches the frontend perfectly
        search_kwargs["filter"] = lambda metadata: os.path.basename(metadata.get("source", "")) in selected_files
        
        
    # Use MMR for diverse results
    retrieved_docs = vectorstore.max_marginal_relevance_search(
        question, 
        **search_kwargs
    )
    
    doc_texts = [doc.page_content for doc in retrieved_docs]
    return {"documents": doc_texts}


# --- NODE 1: THE SCORING EVALUATOR ---
class DocEvaluations(BaseModel):
    scores: list[float] = Field(description="A list of scores, one for each chunk in the exact order provided.")
    optimized_web_query: str = Field(description="If chunks lack info, provide an optimized 6-14 word search query. Else, output empty string.")

def eval_each_doc_node(state: GraphState):
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"good_documents": [], "verdict": "INCORRECT", "reason": "No documents retrieved.", "web_query": question}
        
    chunks_text = ""
    for i, doc in enumerate(documents):
        chunks_text += f"--- Chunk {i+1} ---\n{doc}\n\n"
        
    doc_eval_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an intelligent retrieval evaluator for a RAG system.\n"
         "1. Evaluate EACH chunk individually and return a list of scores in [0.0, 1.0].\n"
         "- Score 1.0: The chunk contains the specific information requested.\n"
         "- Score 0.0: The chunk is completely irrelevant.\n"
         "2. Check for missing information. If the user asks a compound question (e.g., local info AND global info), and the global info is missing from the chunks, generate an optimized web search query for that missing public knowledge.\n"
         "Do NOT search the web for personal details missing from the document. Only search for public/external facts."),
        ("human", "Query: {question}\n\n{chunks}")
    ])
    
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvaluations)
    result = doc_eval_chain.invoke({"question": question, "chunks": chunks_text})
    
    good_docs = []
    scores = result.scores
    
    for i, score in enumerate(scores):
        if i < len(documents) and score > LOWER_TH:
            good_docs.append(documents[i])
            
    has_good_chunk = scores and any(s > UPPER_TH for s in scores)
    needs_web_search = bool(result.optimized_web_query.strip())
    
    if has_good_chunk and not needs_web_search:
        verdict = "CORRECT"
        reason = "Found perfect local data. No web search needed."
    elif has_good_chunk and needs_web_search:
        verdict = "AMBIGUOUS"
        reason = "Found local data, but triggered web search for missing parts of a compound query."
    elif not has_good_chunk:
        verdict = "INCORRECT"
        reason = f"No chunks scored > {UPPER_TH}. Relying on web search."
        
    final_query = result.optimized_web_query if result.optimized_web_query else question
    
    return {"good_documents": good_docs, "verdict": verdict, "reason": reason, "web_query": final_query}


# --- NODE 2: THE QUERY REWRITER ---
def rewrite_query_node(state: GraphState):
    return {}


# --- NODE 3: THE WEB SEARCH ---
def web_search_node(state: GraphState):
    print("---WEB SEARCH---")
    query = state.get("web_query") or state["question"]
    tavily = TavilySearchResults(max_results=2)
    results = tavily.invoke({"query": query})
    
    web_docs = []
    for r in results or []:
        content = r.get("content", "") or r.get("snippet", "")
        web_docs.append(content)
        
    return {"web_documents": web_docs}


# --- NODE 4: THE  REFINER ---
def refine(state: GraphState):
    print("---REFINING KNOWLEDGE ---")
    
    local_docs = state.get("good_documents", [])
    web_docs = state.get("web_documents", [])
    
    context_parts = []
    
    if local_docs:
        context_parts.append("=== INTERNAL DOCUMENT DATA (HIGH PRIORITY) ===")
        context_parts.extend(local_docs)
        context_parts.append("==============================================")
        
    if web_docs:
        context_parts.append("=== EXTERNAL WEB SEARCH DATA (FOR PUBLIC FACTS ONLY) ===")
        context_parts.extend(web_docs)
        context_parts.append("========================================================")
        
    refined_context = "\n\n".join(context_parts).strip()
    
    return {"refined_context": refined_context}


# --- NODE 5: THE FINAL GENERATOR ---
def generate(state: GraphState):
    print("---GENERATING FINAL ANSWER---")
    question = state["question"]
    context = state.get("refined_context", "")
    
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert AI extraction assistant. Your job is to answer the user's question based strictly on the provided context.\n"
         "The context consists of raw, unstructured document chunks. You must synthesize the text to find the answer.\n"
         "If the information is completely missing, say: 'I don't know.'"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    
    generator_chain = answer_prompt | llm
    out = generator_chain.invoke({"question": question, "context": context})
    
    return {"generation": out.content}