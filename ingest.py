import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data/"
DB_PATH = "db/"

def create_vector_db():
    # 1. Load documents (PDF and CSV)
    print("📂 Loading documents from /data...")
    
    # PDF Loader
    pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    # CSV Loader
    csv_loader = DirectoryLoader(DATA_PATH, glob="*.csv", loader_cls=CSVLoader)
    
    docs = pdf_loader.load() + csv_loader.load()
    print(f"✅ Loaded {len(docs)} files.")

    # 2. Split into Chunks (Text Fragmentation)
    # We use chunks of 1000 characters with overlap to avoid losing context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"✂️ Documents split into {len(splits)} fragments.")

    # 3. Create Embeddings (Convert text to numbers)
    print("🧠 Generating embeddings (this might take a while)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Save to ChromaDB
    print(f"💾 Saving to local vector database at: {DB_PATH}")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print("✨ Process completed successfully!")

if __name__ == "__main__":
    # Create data folder if it doesn't exist
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"⚠️ Folder {DATA_PATH} created. Put your files there and run again.")
    else:
        create_vector_db()