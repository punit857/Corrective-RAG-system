# CRAG (Corrective RAG) Engine

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-009688)
![LangGraph](https://img.shields.io/badge/LangGraph-AI-FF9900)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black)

## What is this app?
This is an advanced, Retrieval Augmented Generation (RAG) application implementing an agentic **Corrective RAG (CRAG)** architecture. 

Unlike standard RAG systems that blindly trust document searches, this system acts as a self-correcting agent. It evaluates retrieved data for relevance and autonomously triggers live web searches to supplement its context when internal knowledge is insufficient or ambiguous.



🌐 **Live Demo:** [https://corrective-rag-system.vercel.app](https://corrective-rag-system.vercel.app)

----------------

##  Architecture & Tech Stack

**Frontend:** Vanilla HTML5, JavaScript, TailwindCSS (Deployed on Vercel edge network)

**Backend:** FastAPI, Python (Deployed on Render)

**AI Orchestration:** LangChain & LangGraph

**LLM:** Meta Llama 3 (via Groq API)

**Vector Database:** FAISS

**Embeddings:** FastEmbed 

**Web Search Tool:** Tavily Search API

--------------

##  The Workflow
<img width="553" height="652" alt="workflow" src="https://github.com/user-attachments/assets/cb85c44f-db4f-4155-90f5-55b828f9c7a3" />


When a user submits a question, the application executes a state machine using LangGraph:

- **Document Retrieval & Filtering:** The system converts the user's question into vector embeddings using FastEmbed. It queries the local FAISS database, applying metadata filters to only scan PDF files explicitly selected by the user.

- **Relevance Evaluation:** The LLM reads the retrieved text chunks and scores them based on relevance to the user's query. If the documents contain the necessary information, the state is marked `CORRECT`.If the information is missing, it is marked `INCORRECT`. If partial information is available in documents, the state is marked `AMBIGUOUS`.

- **Query Rewriting:** If the verdict is `INCORRECT` or `AMBIGUOUS`, the system prompts the LLM to rewrite the user's original question into an optimized web search query.

- **Web Search Fallback:** The system uses the Tavily Search API to scrape the live internet for the missing information and adds the top website snippets to the context window.

- **Answer Generation:** The Llama-3 model reads the compiled context (local document chunks + web data) and generates a highly accurate final response.

- **Execution Transparency:** The backend packages the entire LangGraph execution log (node verdicts, generated web queries, and raw text snippets) and sends it to the frontend. Users can view this data to audit the AI's reasoning process.

##  Key Features
- **Explainable AI :** The UI exposes the entire LangGraph execution log, allowing users to audit the exact reasoning steps, generated web queries, and raw text snippets the AI used.
<img width="761" height="375" alt="Screenshot 2026-03-05 190111" src="https://github.com/user-attachments/assets/e5e5774d-2893-4ce4-85ac-cba7f8bfeec1" />


- **Dynamic Metadata Filtering:** Users can upload multiple PDFs and dynamically toggle which specific documents the vector database should scan for any given query.
  <img width="842" height="576" alt="Screenshot 2026-03-05 190016" src="https://github.com/user-attachments/assets/828dcdbe-a384-4663-ac4b-ed01228cb339" />

- **Smart Fallback:** Eliminates hallucinations by autonomously searching the live internet when local documents fall short.
_______________________

## Setup instructions
**1. Clone the Repository & install Dependencies:**
```
git clone [https://github.com/your-username/Corrective-RAG-system.git](https://github.com/your-username/Corrective-RAG-system.git)
cd Corrective-RAG-system
pip install -r requirements.txt
```
**2. Set up Environment Variables:**

Create a .env file in the root directory:
```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```
**3. Run the Backend:**
```
uvicorn api:app --host 127.0.0.1 --port 8000
```

**4. Run the Frontend:**
Open index.html in your browser. (Note: Ensure the API endpoint in the JS fetch requests points to http://127.0.0.1:8000 for local development).
