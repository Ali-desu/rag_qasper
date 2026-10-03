# A small RAG project for research papers

I’m building a local RAG system that answers questions about research papers. I’m using the QASPER dataset, so I can check whether the system finds paragraphs that people marked as evidence for each answer.

The project is still in progress. The retrieval, reranking, and generation pieces are working, and I’m measuring how well they work before treating the whole system as finished.

## What it does

The current pipeline:

1. Loads papers from the QASPER validation dataset.
2. Splits each paper into paragraph-sized chunks and keeps the paper and section names with each chunk.
3. Embeds the chunks with `Snowflake/snowflake-arctic-embed-s` and stores them in SQLite using `sqlite-vec`.
4. Searches with both vector similarity and BM25 keyword search, then combines the ranked results with Reciprocal Rank Fusion.
5. Reranks the best chunks with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
6. Can send the question and retrieved chunks to a local Qwen model through Ollama to generate an answer with section citations.

The search is currently scoped to the paper the question is about. It does not search across the whole collection yet.

## Current results

On the full QASPER validation split, hybrid retrieval did a little better than dense search by itself. For 892 questions, hybrid hit@5 was 0.697, compared with 0.673 for dense search and 0.628 for BM25. At k=20, hybrid found at least one evidence paragraph for about 96% of questions.

In an earlier experiment, I tried `BAAI/bge-reranker-v2-m3` on 20 papers and saw the correct evidence move closer to the top of the results. It took about 50 minutes on my CPU, so the current API uses the smaller `cross-encoder/ms-marco-MiniLM-L-6-v2` model instead. The local Qwen model can generate answers too, but CPU generation is slow and I still need to evaluate how reliably its answers follow the retrieved text.

These are early results from my current setup, not a claim that the system is finished or that every generated answer is correct. I keep more detail about the experiments and decisions in [NOTES.md](NOTES.md).

## Running it

Install the packages in `requirements.txt`. The paper data and SQLite database should be available in `data/`. To download the QASPER validation data if needed, run:

```bash
python scripts/download_qasper.py
```

Prepare the QASPER data and SQLite database, then start Ollama with the model configured in `src/rag/generator.py`. Start the API and UI in separate terminals from the project root:

```bash
uvicorn backend.app.main:app --reload
```

```bash
streamlit run frontend/app.py
```

The app lists papers from the database, shows their abstracts and sections, and sends questions to the API. The API is also available at `http://localhost:8000/docs`.

## How the app works

### Preparing the database

The QASPER validation dataset contains paper metadata and full text grouped into sections and paragraphs. The ingestion process saves each paper's ID, title, and abstract in the `papers` table. The ingestor turns each non-empty paragraph into a chunk and keeps its paper ID, section name, and position. The embedder creates a normalized vector for every chunk.

The data is stored in one SQLite database:

- `papers` stores paper IDs, titles, and abstracts.
- `chunks` stores paragraph text, section names, and ordering metadata.
- `vec_chunks` stores paragraph vectors through `sqlite-vec`.
- `fts_chunks` is an FTS5 full-text index used for BM25 keyword search.

### Reading a paper and asking a question

The Streamlit app first calls `GET /papers` to fill the paper picker. After a paper is selected, it calls `GET /papers/{paper_id}` and displays the abstract and paragraphs grouped by section. This reading view is the full paper text stored in the database.

When the user submits a question, Streamlit sends the question and selected paper ID to `POST /ask`. The API embeds the question, runs dense vector search and BM25 keyword search, and limits both searches to chunks from that paper. Each search contributes up to 20 candidates. Reciprocal Rank Fusion combines the ranked lists using `k=60`, and the API keeps the top 5 chunks.

The cross-encoder reranker scores those 5 question-and-paragraph pairs and orders the chunks again. The API sends the question and reranked paragraph text, with section names, to the Qwen model served locally by Ollama. The model is instructed to answer from those paragraphs, cite section names, and say when the evidence does not contain an answer. FastAPI returns the answer, timing information, and section names to Streamlit.

The model does not receive every paragraph shown in the reading view. It only receives the retrieved and reranked chunks, so an answer can be missed if retrieval does not bring its evidence into those five chunks.

## Project files

- `src/rag/` contains the ingestion, embedding, storage, retrieval, reranking, and generation code.
- `backend/app/main.py` provides the FastAPI endpoints.
- `frontend/app.py` contains the Streamlit interface.
- `scripts/download_qasper.py` downloads the validation split.
- `NOTES.md` has experiment results and implementation decisions.

## Next steps

- Find a smaller reranker that runs faster on CPU and compare its results.
- Evaluate the reranker on more papers and record the metrics and timing.
- Check whether generated answers stay grounded in the retrieved paragraphs.
- Try searching across all papers instead of filtering to the paper in the question.
- Add a simple command-line entry point for preparing the database.
