class Retriever:
    def __init__(self, embedder, store, rrf_k=60, candidates=20):
        self.embedder = embedder
        self.store = store
        self.rrf_k = rrf_k            # the 60 in 1 / (60 + rank)
        self.candidates = candidates  # how many results to take from each method

    def dense(self, question, k=5, paper_id=None):
        query_vector = self.embedder.embed_query(question)
        hits = self.store.dense_search(query_vector, k=k, paper_id=paper_id)
        return [chunk_id for chunk_id, _ in hits]

    def keyword(self, question, k=5, paper_id=None):
        hits = self.store.keyword_search(question, k=k, paper_id=paper_id)
        return [chunk_id for chunk_id, _ in hits]

    def rrf(self, result_lists):
        """Merge several ranked lists of chunk ids into one.

        result_lists: e.g. [[12, 7, 3, ...], [7, 40, 12, ...]], best first in each
        returns: list of chunk ids, best first
        """
        scores = {}
        for ids in result_lists:
            for rank, chunk_id in enumerate(ids, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (self.rrf_k + rank)

        return sorted(scores, key=scores.get, reverse=True)
                

    def hybrid(self, question, k=5, paper_id=None):
        dense_ids = self.dense(question, k=self.candidates, paper_id=paper_id)
        keyword_ids = self.keyword(question, k=self.candidates, paper_id=paper_id)
        return self.rrf([dense_ids, keyword_ids])[:k]

    def search(self, question, k=5, paper_id=None, method="hybrid"):
        """Returns the full chunks (dicts), best first."""
        if method == "dense":
            ids = self.dense(question, k, paper_id)
        elif method == "keyword":
            ids = self.keyword(question, k, paper_id)
        else:
            ids = self.hybrid(question, k, paper_id)
        return self.store.get_chunks(ids)