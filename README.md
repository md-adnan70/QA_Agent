# Document Q&A AI Agent with Arxiv Search

This project implements a sophisticated AI agent capable of answering questions about a collection of PDF documents and dynamically searching the Arxiv repository for academic papers. It's built using a Retrieval-Augmented Generation (RAG) architecture with Python, LangChain, and Google's Gemini Pro LLM.

## Features

-   **Multi-Document Q&A:** Ask questions about a local library of PDF documents.
-   **Content-Aware Responses:** Answers are grounded in the provided documents to prevent hallucination.
-   **Intelligent Tool Use:** The agent can decide whether to query local documents or search Arxiv online based on the user's question.
-   **Specific Functionalities:**
    -   **Direct Lookup:** "What was the conclusion of the 'Attention is All You Need' paper?"
    -   **Summarization:** "Summarize the methodology section of paper X."
    -   **Data Extraction:** "What F1-score did they report for the baseline model?"
-   **Arxiv Search:** "Find me papers on Arxiv about Mixture of Experts models."

## Architecture

The system uses a Retrieval-Augmented Generation (RAG) pipeline for the local document Q&A and a ReAct (Reasoning and Acting) agent to orchestrate tool use.

1.  **Ingestion:** PDFs in the `/data` folder are parsed, chunked, and stored as vector embeddings in a ChromaDB database.
2.  **Agent Logic:** A LangChain agent is equipped with two tools:
    -   `DocumentQA`: A RAG chain that retrieves relevant text from the vector store to answer questions.
    -   `ArxivSearch`: A tool that queries the official Arxiv API.
3.  **Execution:** The agent analyzes the user's input and intelligently routes the request to the appropriate tool to generate the final answer.

## Setup & Installation

### Prerequisites

-   Python 3.9+
-   A Google AI API Key

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd document-qna-agent
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
    -   Add your Google API key to it:
        ```
        GOOGLE_API_KEY="your_google_api_key_here"
        ```

## Usage

1.  **Add Documents:** Place your PDF files into the `/data` directory.

2.  **Run the Ingestion Pipeline (One-time step):**
    This will process your PDFs and create the vector store.
    ```bash
    python ingest.py
    ```

3.  **Start the Q&A Agent:**
    Now you can start asking questions.
    ```bash
    python agent.py
    ```
    -   Follow the on-screen prompt to interact with the agent.
    -   Type `exit` to end the session.