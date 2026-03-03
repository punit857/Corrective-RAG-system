import os
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import crag_app 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from nodes import vectorstore 

app = FastAPI(
    title="CRAG API",
    description="A Corrective RAG API with dynamic web-search fallback.",
    version="1.1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    selected_files: list[str] = []

class WorkflowStep(BaseModel):
    node_executed: str
    verdict: str | None = None
    generated_web_query: str | None = None
    used_documents: list[str] | None = None  

class QueryResponse(BaseModel):
    final_answer: str
    workflow_log: list[WorkflowStep]

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        inputs = {"question": request.question, "selected_files": request.selected_files}
        workflow_log = []
        final_answer = ""
        
        for output in crag_app.stream(inputs):
            for node_name, state_data in output.items():
                if not isinstance(state_data, dict):
                    state_data = {}
                
                
                docs_to_log = []
                if node_name in ["retrieve_node", "retrieve"]:
                    docs_to_log = state_data.get("documents", [])
                elif node_name in ["eval_each_doc_node", "eval_each_doc"]:
                    docs_to_log = state_data.get("good_documents", [])
                elif node_name in ["web_search_node", "web_search"]:
                    docs_to_log = state_data.get("web_documents", [])
                elif node_name == "refine":
                    ref_ctx = state_data.get("refined_context")
                    if ref_ctx: docs_to_log = [ref_ctx]
                # ---------------------------------------------------
                
                step_info = WorkflowStep(
                    node_executed=node_name,
                    verdict=state_data.get("verdict"),
                    generated_web_query=state_data.get("web_query"),
                    used_documents=docs_to_log if docs_to_log else None 
                )
                workflow_log.append(step_info)
                
                if node_name == "generate":
                    final_answer = state_data.get("generation", "Error generating answer.")
                    
        return QueryResponse(final_answer=final_answer, workflow_log=workflow_log)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs("documents", exist_ok=True)
        file_path = f"documents/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        new_chunks = text_splitter.split_documents(docs)
        
        for d in new_chunks:
            clean_text = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
            source_file = os.path.basename(d.metadata.get("source", file.filename))
            d.page_content = f"[Source Document: {source_file}]\n{clean_text}"
            
        vectorstore.add_documents(new_chunks)
        return {"filename": file.filename, "status": "Uploaded and indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    os.makedirs("documents", exist_ok=True)
    files = [f for f in os.listdir("documents") if f.endswith(".pdf")]
    return {"documents": files}


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    try:
        file_path = f"documents/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"status": "Deleted successfully from disk."}
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "Active"}

   