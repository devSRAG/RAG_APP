from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils.vectorstore import retrieve_context
from utils.llm_reasoning import generate_answer
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

class QueryInput(BaseModel):
    document: str
    question: str

@app.post("/rag")
def rag_answer(data: QueryInput):
    context = retrieve_context(data.document, data.question)
    answer = generate_answer(context, data.question)
    return {
        "question": data.question,
        "answer": answer,
        "context": context
    }
