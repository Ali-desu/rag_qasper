from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag.generator import Generator
from src.rag.retriever import Retriever
from src.rag.vector_db import VectorStore
from src.rag.embedder import Embedder
from src.rag.reranker import Reranker

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    paper_id: str

generator = Generator()
retriever = Retriever(embedder=Embedder(), store=VectorStore())


@lru_cache(maxsize=1)
def get_reranker():
    """Load the reranker once and reuse it for later requests."""
    return Reranker()


@app.get("/papers")
async def list_papers():
    """List papers available in the database."""
    return retriever.store.get_papers()


@app.get("/papers/{paper_id}")
async def get_paper(paper_id: str):
    """Return a paper's metadata and paragraphs in reading order."""
    paper = next(
        (paper for paper in retriever.store.get_papers() if paper["paper_id"] == paper_id),
        None,
    )
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    paper["sections"] = retriever.store.get_paper_chunks(paper_id)
    return paper


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    paper_id = request.paper_id

    # Retrieve relevant chunks from the vector store
    chunks = retriever.search(question, k=5, paper_id=paper_id)

    # Rerank the chunks
    chunks = get_reranker().rerank(question, chunks, 5)

    # Generate an answer using the retrieved chunks
    answer, stats = generator.generate(question, chunks)

    return {
        "answer": answer,
        "stats": stats,
        "sections_used": [c["section"] for c in chunks]
    }
