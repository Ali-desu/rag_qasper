from fastapi import FastAPI
from pydantic import BaseModel
from src.rag.generator import Generator
from src.rag.retriever import Retriever
from src.rag.vector_db import VectorStore
from src.rag.embedder import Embedder


app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    paper_id: str

generator = Generator()
retriever = Retriever(embedder=Embedder(), store=VectorStore())


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    paper_id = request.paper_id

    # Retrieve relevant chunks from the vector store
    chunks = retriever.search(question, k=3, paper_id=paper_id)

    # Generate an answer using the retrieved chunks
    answer, stats = generator.generate(question, chunks)

    return {
        "answer": answer,
        "stats": stats,
        "sections_used": [c["section"] for c in chunks]
    }
