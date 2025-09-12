# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
load_dotenv()
PDF_SOURCE_DIR = "F:/AI_Projects/Tasks/QA_Agent/data"
VECTOR_STORE_DIR = "F:/AI_Projects/Tasks/QA_Agent/vector_store"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def main():
    """
    Main function to ingest PDF documents.
    - Loads documents from the source directory.
    - Splits documents into manageable chunks.
    - Creates embeddings for the chunks.
    - Stores the embeddings in a Chroma vector store.
    """
    print("🚀 Starting document ingestion process...")

    # Load all PDF documents from the specified directory
    documents = []
    for filename in os.listdir(PDF_SOURCE_DIR):
        if filename.endswith('.pdf'):
            filepath = os.path.join(PDF_SOURCE_DIR, filename)
            try:
                loader = PyMuPDFLoader(filepath)
                documents.extend(loader.load())
                print(f"✅ Loaded {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

    if not documents:
        print("No documents found. Exiting.")
        return

    # Split the documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(docs)} chunks.")

    # Initialize the embedding model (using a local, open-source model)
    # This is great for privacy and cost-effectiveness.
    print("Initializing embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have a GPU
    )

    # Create and persist the vector store
    print("Creating and persisting vector store...")
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=VECTOR_STORE_DIR
    )

    print("✅ Ingestion complete! Vector store created at:", VECTOR_STORE_DIR)

if __name__ == "__main__":
    main()