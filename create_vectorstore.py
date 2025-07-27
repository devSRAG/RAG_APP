from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import os
import time

# 1️⃣ Load documents
loader = TextLoader("data.txt")  # replace with your actual data file
documents = loader.load()

# 2️⃣ Split text
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3️⃣ Embedding with caching
start = time.time()
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    cache_folder="./hf_cache"
)


print(f"⏱️ Embedding loaded in {time.time() - start:.2f}s")

# 4️⃣ Save to vector DB
persist_directory = "vectordb"
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
vectordb.persist()

print("✅ Vectorstore created and persisted.")
