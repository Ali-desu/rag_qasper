# A small RAG project for research papers

I’m building a local RAG system that answers questions about research papers. I’m using the QASPER dataset, so I can check whether the system finds paragraphs that people marked as evidence for each answer.

The project is still in progress. The retrieval, reranking, and generation pieces are working, and I’m measuring how well they work before treating the whole system as finished.

## What it does

The current pipeline:

1. Loads papers from the QASPER validation dataset.
2. Splits each paper into paragraph-sized chunks and keeps the paper and section names with each chunk.
3. Embeds the chunks with `Snowflake/snowflake-arctic-embed-s` and stores them in SQLite using `sqlite-vec`.
4. Searches with both vector similarity and BM25 keyword search, then combines the ranked results with Reciprocal Rank Fusion.
5. Can rerank retrieved chunks with `BAAI/bge-reranker-v2-m3`.
6. Can send the question and retrieved chunks to a local Qwen model through Ollama to generate an answer with section citations.

The search is currently scoped to the paper the question is about. It does not search across the whole collection yet.

## Current results

On the full QASPER validation split, hybrid retrieval did a little better than dense search by itself. For 892 questions, hybrid hit@5 was 0.697, compared with 0.673 for dense search and 0.628 for BM25. At k=20, hybrid found at least one evidence paragraph for about 96% of questions.

I also tried the reranker on 20 papers and saw the correct evidence move closer to the top of the results. It took about 50 minutes on my CPU, though, so I’m looking at smaller reranker models. The local Qwen model can generate answers too, but CPU generation is slow and I still need to evaluate how reliably its answers follow the retrieved text.

These are early results from my current setup, not a claim that the system is finished or that every generated answer is correct. I keep more detail about the experiments and decisions in [NOTES.md](NOTES.md).

## Running it

Install the packages in `requirements.txt`. The paper data and SQLite database should be available in `data/`. To download the QASPER validation data if needed, run:

```bash
python scripts/download_qasper.py
```

The notebook in `notebooks/explore.ipynb` is used to build the database. Start Ollama with the model configured in `src/rag/generator.py`, then start the API and UI in separate terminals from the project root:

```bash
uvicorn backend.app.main:app --reload
```

```bash
streamlit run frontend/app.py
```

The app lists papers from the database, shows their abstracts and sections, and sends questions to the API. The API is also available at `http://localhost:8000/docs`.

## Project files

- `src/rag/` contains the ingestion, embedding, storage, retrieval, reranking, and generation code.
- `backend/app/main.py` provides the FastAPI endpoints.
- `frontend/app.py` contains the Streamlit interface.
- `scripts/download_qasper.py` downloads the validation split.
- `notebooks/explore.ipynb` is the notebook for trying the pipeline and preparing the database.
- `NOTES.md` has experiment results and implementation decisions.

## Next steps

- Find a smaller reranker that runs faster on CPU and compare its results.
- Evaluate the reranker on more papers and record the metrics and timing.
- Check whether generated answers stay grounded in the retrieved paragraphs.
- Try searching across all papers instead of filtering to the paper in the question.
- Add a simple command-line entry point so it is easier to run without the notebook.
