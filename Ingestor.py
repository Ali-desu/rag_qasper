import time

from datasets import load_dataset


class Ingestor:
    def __init__(self, data_path="data/qasper_validation.parquet"):
        self.data_path = data_path

    def load_qasper_validation(self):
        return load_dataset("parquet", data_files=self.data_path, split="train")

    def chunk_data(self, dataset):
        # each paragraph becomes one chunk, with its paper, section and position
        chunked_data = []

        for paper in dataset:
            paper_id = paper["id"]
            text = paper["full_text"]
            index = 0

            for section, paragraphs in zip(text["section_name"], text["paragraphs"]):
                for p in paragraphs:
                    # skip empty paragraphs
                    if not p.strip():
                        continue
                    chunked_data.append({
                        "id": f"{paper_id}::{index}",
                        "paper_id": paper_id,
                        "section": section,
                        "chunk_index": index,
                        "text": p,
                    })
                    index += 1

        return chunked_data

    def ingest(self, embedder, store, max_papers=None):
        """Load, chunk, embed and store everything (rebuilds the database from scratch).

        max_papers: only ingest the first N papers, useful for quick tests.
        """
        dataset = self.load_qasper_validation()
        if max_papers is not None:
            dataset = dataset.select(range(max_papers))

        chunked_data = self.chunk_data(dataset)
        print(f"{len(dataset)} papers -> {len(chunked_data)} chunks")

        start = time.perf_counter()
        vectors = embedder.embed_documents([c["text"] for c in chunked_data])
        print(f"Embedded in {time.perf_counter() - start:.1f}s")

        store.reset()
        store.add_chunks(chunked_data, vectors)

        counts = store.count()
        print(f"Stored: {counts}")
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Tables out of sync: {counts}")

        return chunked_data