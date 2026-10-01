# Notes

Running log of decisions, experiments and results. Newest results at the bottom of each section.

## Setup

- **Data:** QASPER validation split (281 NLP papers, questions written by people who only read the title and abstract, answers backed by evidence paragraphs).
- **Chunking:** one chunk per paragraph, empty paragraphs skipped. 13,594 chunks, about 48 per paper.
- **Embeddings:** `Snowflake/snowflake-arctic-embed-s` (33M params, 384 dims, 512 max tokens). Queries use the model's `query` prompt, documents get none. Vectors are normalized.
- **Storage:** SQLite, one file, three tables sharing the same id:
  - `chunks` for text and metadata
  - `vec_chunks` (sqlite-vec) for dense vectors
  - `fts_chunks` (FTS5, porter stemming) for BM25
- **Hybrid search:** dense and BM25 each return 20 candidates, merged with Reciprocal Rank Fusion (k = 60).
- **Search scope:** within the paper the question is about (`paper_id` filter). That's how QASPER questions were written ("which datasets did *they* use?").

## Decisions and why

- **Paragraph chunks.** QASPER evidence is always a full paragraph, so a chunk is correct if its text exactly matches an evidence paragraph. No fuzzy matching needed.
- **arctic-embed-s over bge-small.** Almost the same retrieval score on MTEB (51.98 vs 51.68), well documented, widely used, fast on CPU. Kept as the baseline; bigger models come later as experiments.
- **Did not pick Yuan-embedding-2.0-en** even though it ranked first on MTEB retrieval: 0.6B params, query format not documented, training data not disclosed, very few downloads.
- **SQLite instead of a vector DB server.** No Docker, zero setup, BM25 and vector search in one file. Exact search is fine at this size. In production I'd move to pgvector or Qdrant.
- **Wrote TF-IDF by hand first** to understand sparse retrieval before using FTS5's BM25.

## Evaluation

For each answerable question (tables and figures excluded), I record the ranks where evidence paragraphs appear, then compute:

- **hit@k:** at least one evidence paragraph in the top k
- **MRR@k:** 1 / rank of the first evidence paragraph (0 if none)
- **recall@k:** evidence paragraphs found / evidence paragraphs that exist
- **precision@k:** evidence paragraphs found / k

Metrics are averaged over questions, not papers, so every question counts once.

Things to keep in mind when reading the numbers:

- Evaluation is strict. Only paragraphs the annotators picked count as correct, so some "misses" are actually fine answers.
- Precision has a low ceiling: most questions have one or two evidence paragraphs, so precision@5 can't go much above 0.2 to 0.4.
- Search is within one paper of about 48 chunks, so large k values (like 20) cover a big part of the paper. hit@1 and hit@5 are the meaningful ones.

## Results

### 2026-09-27: first comparison, 1 paper, 4 questions

| method | hit@5 | MRR |
|---|---|---|
| dense | 0.75 | 0.625 |
| TF-IDF (mine) | 0.50 | 0.312 |

Way too small to conclude anything, but useful to catch bugs. Noticed that the Acknowledgments section ranked first for "which datasets did they experiment with?". Embeddings match the topic loosely, not the answer.

### 2026-09-29: 10 papers, 29 questions

| method | hit@5 | MRR | recall@5 |
|---|---|---|---|
| dense | 0.724 | 0.506 | 0.459 |
| keyword (BM25) | 0.690 | 0.366 | 0.373 |
| hybrid | 0.724 | 0.493 | 0.440 |

Here hybrid did not help. But each question is worth 3.4 points of hit@5, so this is noise.

### 2026-09-29: full validation set, 281 papers, 892 questions

| method | k | hit | MRR | recall | precision |
|---|---|---|---|---|---|
| dense | 1 | 0.287 | 0.287 | 0.195 | 0.287 |
| | 5 | 0.673 | 0.428 | 0.515 | 0.167 |
| | 10 | 0.825 | 0.448 | 0.679 | 0.117 |
| | 20 | 0.938 | 0.457 | 0.833 | 0.076 |
| keyword | 1 | 0.224 | 0.224 | 0.155 | 0.224 |
| | 5 | 0.628 | 0.371 | 0.477 | 0.154 |
| | 10 | 0.796 | 0.394 | 0.652 | 0.111 |
| | 20 | 0.920 | 0.402 | 0.815 | 0.073 |
| **hybrid** | 1 | **0.290** | **0.290** | **0.196** | **0.290** |
| | 5 | **0.697** | **0.442** | **0.537** | **0.174** |
| | 10 | **0.836** | **0.460** | **0.693** | **0.118** |
| | 20 | **0.957** | **0.469** | **0.855** | **0.077** |

What I take from it:

- **Hybrid wins at every k**, and dense beats BM25 at every k. At k = 5, hybrid finds the answer for about 21 more questions than dense alone. The 10-paper result was just too small a sample.
- **Ranking is the bottleneck, not finding.** The answer is in the top 20 for 96% of questions, but in the top 5 for only 70%, and first for only 29%. About 230 questions have the right paragraph somewhere between rank 6 and 20.
- **That's the case for a reranker.** Rerank the top 20 and the ceiling for hit@5 is around 96%.
- About 4% of questions (~40) have no evidence even in the top 20. Those are real retrieval failures that a reranker can't fix.


### 2026-10-1: added reranker and generator

- **BAAI/bge-reranker-v2-m3 reranker** added, I tested it on 20 papers and it improved the results. the correct results are more likely to appear at the top of the list , the only problem is that the model is very slow on cpu , took 50 minutes on 20 papers . i will probably change it to another smaller reranker

- **qwen model** works good but is also pretty slow in cpu , and i should add another model in order to approve of the model's results and see if they are correct and are using the chunks retrieved

overall we do have now working components of an end to end rag system.