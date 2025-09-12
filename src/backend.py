# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
from agent import initialize_llm, create_rag_retriever, create_rag_chain, setup_arxiv_tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# This dictionary will hold our "global" state, like the agent executor
app_state = {}

# --- Lifespan Management for the App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model and agent
    print("🚀 Server starting up...")
    print("🤖 Initializing AI Agent... (This may take a moment)")
    
    llm = initialize_llm()
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    rag_retriever = create_rag_retriever(embedding_model)
    rag_chain = create_rag_chain(llm, rag_retriever)
    arxiv_tool = setup_arxiv_tool()

    tools = [
    Tool(
        name="DocumentQA",
        func=rag_chain.invoke,
        description="Use this tool to answer questions about the content of the uploaded PDF documents. This should be your default choice for questions about methodology, conclusions, or specific results from the documents."
    ),
    arxiv_tool, # This is a Tool object
]
    
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    app_state["agent_executor"] = agent_executor
    print("✅ Agent is ready!")
    
    yield
    
    # Shutdown: Clean up resources if needed
    print("👋 Server shutting down...")
    app_state.clear()


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Q&A Agent API"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Receives a user query, processes it with the agent, and returns the answer.
    """
    print(f"Received query: {request.query}")
    agent_executor = app_state.get("agent_executor")
    if not agent_executor:
        return {"answer": "Error: Agent not initialized. Please restart the server."}

    try:
        response = await agent_executor.ainvoke({"input": request.query})
        return {"answer": response.get("output", "No response generated.")}
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        return {"answer": f"An error occurred: {e}"}

if __name__ == "__main__":
    # Note: For production, you'd use a proper ASGI server like Gunicorn with Uvicorn workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)