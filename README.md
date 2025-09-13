# Document Q&A AI Agent Web Application

This project implements a sophisticated AI agent, delivered as a web application, capable of answering questions about a collection of dynamically uploaded PDF documents and searching the Arxiv repository for academic papers. It's built on a modern two-tier architecture using Python, FastAPI for the backend, Streamlit for the frontend, LangChain, and Google's Gemini Pro LLM.

## Features

-   **Dynamic Multi-Document Q&A:** Upload your PDF documents directly in the app and start asking questions.
-   **Content-Aware Responses:** Answers are grounded in the provided documents to prevent hallucination.
-   **Intelligent Tool Use:** The agent decides whether to query your uploaded documents or search Arxiv online.
-   **Specific Functionalities:**
    -   **Direct Lookup:** "What was the conclusion of the 'Attention is All You Need' paper?"
    -   **Summarization:** "Summarize the methodology section of paper X."
    -   **Data Extraction:** "What F1-score did they report for the baseline model?"
-   **Arxiv Search:** "Find me papers on Arxiv about Mixture of Experts models."

---

## Architecture

The system is built on a modern two-tier architecture, separating the user interface from the core AI logic.

1.  **Ingestion:** PDFs in the `/data` folder are parsed, chunked, and stored as vector embeddings in a ChromaDB database.

2.  **Agent Logic:** A LangChain agent is equipped with two tools:

    -   `DocumentQA`: A RAG chain that retrieves relevant text from the vector store to answer questions.

    -   `ArxivSearch`: A tool that queries the official Arxiv API.

3.  **FastAPI Backend:** A robust server that exposes API endpoints for file uploading and querying. It handles all core logic, including:
    * Processing uploaded PDFs on-the-fly.
    * Creating session-specific, in-memory vector stores using chromadb to ensure user data is isolated.
    * Dynamically creating a LangChain agent for each query with the appropriate tools (`DocumentQA` and `ArxivSearch`).

4.  **Streamlit Frontend:** An interactive and user-friendly web interface that allows users to:
    * Upload PDF documents through a simple file-picker.
    * Engage in a conversation with the AI agent through a chat window.
    * Communicate with the FastAPI backend via HTTP requests.

---

## Setup & Installation

### Prerequisites

-   Python 3.10
-   A Groq Inference API key

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/md-adnan70/QA_Agent.git
    cd QA_Agent
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    -   Create a file named `.env` in the root directory.
    -   Add your GROW API key to it:
        ```
        GROQ_API_KEY="your_google_api_key_here"
        ```

---

## Usage

This application runs as two separate services: a backend API and a frontend web app. You will need **two separate terminals** to run them.

### Step 1: Start the Backend Server

In your first terminal, run the FastAPI server. This will handle all the AI processing.

```bash
uvicorn backend:app --reload
```

### Step 2: Start the frontend app

In your second terminal, run the Streamlit app.

```bash
streamlit run app.py
```