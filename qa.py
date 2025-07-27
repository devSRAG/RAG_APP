# rag.py

from langchain_groq import ChatGroq

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# STEP 1: Load environment
load_dotenv()

# STEP 2: Load vector DB
persist_dir = "db"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# STEP 3: Load LLM (Groq Mixtral)
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model=os.getenv("GROQ_MODEL", "llama3-70b-8192")  # fallback if env missing
)

# STEP 4: Custom Prompt
custom_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant answering user queries from the document.

Answer only based on the document context provided. Don't say "according to the document" or "yes, it is mentioned". Instead, speak directly and confidently as if you read and understood it fully.

Give concise, helpful, human-like answers.

Context:
{context}

Question:
{question}
""")

# STEP 5: Context Compression
retriever = vectorstore.as_retriever(search_type="mmr", k=4)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# STEP 6: QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=compression_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# STEP 7: Ask Question
while True:
    query = input("\n🔎 Ask a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})
    print("\n💡 Answer:\n", result["result"])
