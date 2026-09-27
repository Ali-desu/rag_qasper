from Ingestor import Ingestor
from embedder import Embedder
from tfidf_index import TfidfIndex


def get_evidence(answers):
    """All text evidence paragraphs for one question, from every annotator."""
    evidence = set()
    for answer in answers["answer"]:
        for e in answer["evidence"]:
            if not e.startswith("FLOAT SELECTED"):
                evidence.add(e)
    return evidence


def first_hit_rank(top_indices, chunks, evidence):
    """Rank (1-based) of the first correct chunk in the top results, or None."""
    for rank, i in enumerate(top_indices, start=1):
        if chunks[i]["text"] in evidence:
            return rank
    return None


def print_results(name, top_indices, scores, chunks, evidence):
    print(f"  [{name}]")
    for rank, i in enumerate(top_indices, start=1):
        chunk = chunks[i]
        hit = "✅" if chunk["text"] in evidence else "  "
        print(f"    {rank}. {hit} {scores[i]:.3f}  [{chunk['section']}]  {chunk['text'][:70]}...")


def summarize(name, ranks):
    found = [r for r in ranks if r is not None]
    recall = len(found) / len(ranks)
    mrr = sum(1 / r for r in found) / len(ranks)
    print(f"{name:<10} recall@5 = {recall:.2f}   MRR = {mrr:.3f}")


def main():
    ingestor = Ingestor()
    dataset = ingestor.load_qasper_validation()
    chunked_data = ingestor.chunk_data(dataset)

    # chunks of the first paper
    paper = dataset[0]
    paper_chunks = [c for c in chunked_data if c["paper_id"] == paper["id"]]
    texts = [c["text"] for c in paper_chunks]
    print(f"Paper {paper['id']}: {len(paper_chunks)} chunks")

    # build both indexes on the same chunks
    embedder = Embedder()
    chunk_vectors = embedder.embed_documents(texts)

    tfidf = TfidfIndex()
    tfidf.compute_idf(texts)

    dense_ranks, tfidf_ranks = [], []

    qas = paper["qas"]
    for question, answers in zip(qas["question"], qas["answers"]):
        evidence = get_evidence(answers)
        if not evidence:
            continue  # unanswerable or table-only

        print(f"\nQ: {question}")

        # dense (embeddings)
        dense_scores = chunk_vectors @ embedder.embed_query(question)
        dense_top = dense_scores.argsort()[::-1][:5]
        print_results("dense", dense_top, dense_scores, paper_chunks, evidence)

        # sparse (TF-IDF)
        tfidf_scores = tfidf.score(question)
        tfidf_top = [i for i, _ in tfidf.search(question, k=5)]
        print_results("tf-idf", tfidf_top, tfidf_scores, paper_chunks, evidence)

        dense_ranks.append(first_hit_rank(dense_top, paper_chunks, evidence))
        tfidf_ranks.append(first_hit_rank(tfidf_top, paper_chunks, evidence))

    print(f"\n=== {len(dense_ranks)} questions ===")
    summarize("dense", dense_ranks)
    summarize("tf-idf", tfidf_ranks)


if __name__ == "__main__":
    main()