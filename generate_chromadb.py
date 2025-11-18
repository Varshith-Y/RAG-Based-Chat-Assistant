import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ----------------------------------------------------------------------
# Basic configuration
# ----------------------------------------------------------------------

PERSIST_DIRECTORY = "chromadb"  # This folder will be created/overwritten locally


def load_scienceqa_data() -> List[Document]:
    """Load and convert your science QA data into a list of LangChain Documents.

    ⚠️ IMPORTANT:
    Replace this placeholder with your actual data loading logic.
    For example, you might:
      - read a JSON/CSV file
      - loop over question/context/answer fields
      - create Document(page_content=..., metadata={...})

    For now, this returns a tiny dummy example so the script runs.
    """
    # TODO: replace with real ScienceQA data
    texts = [
        "What is photosynthesis? Photosynthesis is the process by which green plants convert light energy into chemical energy.",
        "Explain Newton's first law. An object in motion stays in motion unless acted on by an external force.",
    ]
    return [Document(page_content=t) for t in texts]


def build_chroma_db():
    # Read OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    print("Loading data...")
    docs = load_scienceqa_data()
    print(f"Loaded {len(docs)} documents")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Build and persist Chroma DB
    print("Building Chroma vector store...")
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    vectordb.persist()
    print(f"ChromaDB created and saved to '{PERSIST_DIRECTORY}/'") 


if __name__ == "__main__":
    build_chroma_db()
