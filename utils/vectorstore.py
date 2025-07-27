# utils/vectorstore.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import shutil
import os

# STEP 1: Clear old vector store
persist_dir = "db"
try:
    shutil.rmtree(persist_dir)
    print("✅ Vector DB cleaned")
except PermissionError as e:
    print(f"⚠️ Cannot delete vectorstore: {e}")
except FileNotFoundError:
    pass

# STEP 2: Load PDF
loader = PyMuPDFLoader("data/physics.txt")  # replace with your file
documents = loader.load()

# STEP 3: Split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# STEP 4: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STEP 5: Store vectors
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
vectordb.persist()
print("✅ Embeddings stored successfully")
