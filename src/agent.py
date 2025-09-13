# agent.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import arxiv

# --- Configuration ---
load_dotenv()
VECTOR_STORE_DIR = "F:/AI_Projects/Tasks/QA_Agent/vector_store"

def initialize_llm():
    """Initializes the Gemini language model."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key = os.getenv("GROQ_API_KEY")
    )

def create_rag_retriever(embedding_model):
    """Creates a retriever from the persistent vector store."""
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embedding_model
    )
    return vector_store.as_retriever(search_kwargs={"k": 3})

def create_rag_chain(llm, retriever):
    """Creates the RAG chain for answering questions based on documents."""
    template = """
    SYSTEM: You are a helpful assistant. Use the following context to answer the user's question.
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    Be concise and professional.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def setup_arxiv_tool():
    """Sets up the Arxiv API search tool for the agent."""
    arxiv_search = arxiv.Search(
        query="au:Geoffrey Hinton", # Dummy query to initialize
        max_results=3,
        sort_by=arxiv.SortCriterion.Relevance
    )
    def arxiv_lookup(query: str) -> str:
        """Looks up a paper on Arxiv based on a description or title."""
        try:
            search = arxiv.Search(query=query, max_results=3)
            results = []
            for result in search.results():
                results.append(f"Title: {result.title}\nAuthors: {', '.join(str(a) for a in result.authors)}\nPublished: {result.published.date()}\nSummary: {result.summary[:500]}...\nLink: {result.entry_id}")
            return "\n\n---\n\n".join(results) if results else "No papers found on Arxiv for that query."
        except Exception as e:
            return f"An error occurred with the Arxiv API: {e}"

    return Tool(
        name="ArxivSearch",
        func=arxiv_lookup,
        description="Use this tool to find academic papers on the Arxiv repository when a user asks about a paper you cannot find in the local documents. Input should be a search query like a paper title or topic."
    )

