from Ingestor import Ingestor
from embedder import Embedder


def main():
    ingestor = Ingestor()
    dataset = ingestor.load_qasper_validation()
    chunked_data = ingestor.chunk_data(dataset)

    print(f"Loaded {len(dataset)} papers")
    print(f"Chunked into {len(chunked_data)} chunks")

    embedder = Embedder()

    # chunks of the first paper (chunks are in the same order as the dataset)
    paper_id = chunked_data[0]["paper_id"]
    first_paper_chunks = []
    for chunk in chunked_data:
        if chunk["paper_id"] == paper_id:
            first_paper_chunks.append(chunk)
        else:
            break
    print(f"Paper {paper_id}: {len(first_paper_chunks)} chunks")

    # embed all chunks of this paper
    chunk_vectors = embedder.embed_documents([c["text"] for c in first_paper_chunks])

    # the same paper in the dataset, to get its questions
    paper = dataset[0]
    assert paper["id"] == paper_id

    qas = paper["qas"]
    for question, answers in zip(qas["question"], qas["answers"]):
        # collect the evidence paragraphs from all annotators, skipping tables/figures
        evidence = set()
        for answer in answers["answer"]:
            for e in answer["evidence"]:
                if not e.startswith("FLOAT SELECTED"):
                    evidence.add(e)

        if not evidence:
            continue  # unanswerable or table-only question: skip for now

        query_vector = embedder.embed_query(question)
        scores = chunk_vectors @ query_vector
        top5 = scores.argsort()[::-1][:5]

        print(f"\nQ: {question}")
        for rank, i in enumerate(top5, start=1):
            chunk = first_paper_chunks[i]
            hit = "✅" if chunk["text"] in evidence else "  "
            print(f"  {rank}. {hit} {scores[i]:.3f}  [{chunk['section']}]  {chunk['text'][:90]}...")


if __name__ == "__main__":
    main()