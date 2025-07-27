import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq  # ✅ Correct import
from typing import List

# Load env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not set in .env file!")

# Initialize Groq chat model
llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question.
    If you don't know the answer, say so clearly.

    Context:
    {context}

    Question: {question}
    Answer:"""
)

def answer_with_llm(docs: List[Document], question: str) -> str:
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = (
        {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )
    return chain.invoke(question)
