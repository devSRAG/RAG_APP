import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
persist_dir = "chroma_db"
source_docs_dir = "data"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_cache_dir = "./hf_cache"

# Set environment variable for caching
os.environ["HF_HOME"] = hf_cache_dir

def load_documents():
    texts = []
    for filename in os.listdir(source_docs_dir):
        path = os.path.join(source_docs_dir, filename)
        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(Document(page_content=f.read(), metadata={"source": filename}))
    return texts

def main():
    print("⚙️  Starting pipeline...\n")

    timings = {}

    # Step 1: Load or Create Vectorstore
    t0 = time.time()
    if os.path.exists(persist_dir):
        print("✅ Loaded existing vectorstore.")
        # Load vectorstore without reloading embeddings (faster)
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            cache_folder=hf_cache_dir
        )
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        print("📁 Creating new vectorstore...")
        t_embed = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            cache_folder=hf_cache_dir
        )
        timings["embedding_load"] = time.time() - t_embed

        docs = load_documents()
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()

    timings["vectorstore_load"] = time.time() - t0

    # Step 2: Load LLM
    t0 = time.time()
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192")
    )
    timings["llm_load"] = time.time() - t0

    # Step 3: Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever()
    )

    # Step 4: Ask a question
    query = "Who developed the theory of relativity?"
    t0 = time.time()
    result = qa_chain.invoke({"query": query})
    timings["search_and_answer"] = time.time() - t0

    # Output
    print("\n🧠 Answer:", result["result"])
    print("\n⏱️  Timing Report:")
    if "embedding_load" in timings:
        print(f" - Embedding load:     {timings['embedding_load']:.2f}s")
    print(f" - Vectorstore load:   {timings['vectorstore_load']:.2f}s")
    print(f" - LLM load:           {timings['llm_load']:.2f}s")
    print(f" - Search + Answer:    {timings['search_and_answer']:.2f}s")
    print(f" - Total time:         {sum(timings.values()):.2f}s")

if __name__ == "__main__":
    main()
