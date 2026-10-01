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

The project is still mostly a set of Python modules and a notebook, so there isn’t a one-command app entry point yet. The notebook, `explore.ipynb`, is the place to explore the current pipeline.

To prepare the data, install the packages in `requirements.txt`, then run:

```bash
python download_qasper.py
```

This downloads the QASPER validation data into `data/qasper_validation.parquet`. The notebook and `Ingestor.py` use that file to build the chunks and database. Embedding models are downloaded by Sentence Transformers the first time they are used. To generate answers, Ollama must be running locally with the model configured in `generator.py` available.

## Project files

- `Ingestor.py` loads QASPER and turns paper paragraphs into chunks.
- `embedder.py` creates document and query embeddings.
- `vector_db.py` stores chunks, vectors, and the full-text search index in SQLite.
- `tfidf_index.py` is a small TF-IDF implementation I wrote while learning about sparse retrieval.
- `retriever.py` runs dense, keyword, or hybrid search.
- `reranker.py` reranks retrieved chunks with a cross-encoder.
- `generator.py` calls Ollama with the question and retrieved chunks.
- `download_qasper.py` downloads the validation split.
- `explore.ipynb` is the current notebook for trying the pieces together.
- `NOTES.md` has experiment results and implementation decisions.

## Next steps

- Find a smaller reranker that runs faster on CPU and compare its results.
- Evaluate the reranker on more papers and record the metrics and timing.
- Check whether generated answers stay grounded in the retrieved paragraphs.
- Try searching across all papers instead of filtering to the paper in the question.
- Add a simple command-line entry point so it is easier to run without the notebook.
