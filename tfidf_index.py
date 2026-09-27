import math
import re
from collections import Counter


class TfidfIndex:

    def __init__(self):
        self.idf_values = {}      # token -> idf, computed once
        self.tokenized_docs = []  # tokens of each document
        self.vectors = []         # one normalized sparse vector {token: weight} per document

    def tokenize(self, text):
        # Tokenize the text into words using a simple regex
        return re.findall(r'\b\w+\b', text.lower())

    def compute_idf(self, documents):
        # reset, in case the index is built a second time
        self.idf_values = {}
        self.tokenized_docs = []
        self.vectors = []

        num_documents = len(documents)
        tokens = {}  # token -> how many documents contain this token

        for document in documents:
            tokenized = self.tokenize(document)
            self.tokenized_docs.append(tokenized)
            unique_tokens = set(tokenized)
            for token in unique_tokens:
                tokens[token] = tokens.get(token, 0) + 1

        # now we calculate idf
        for token in tokens:
            self.idf_values[token] = math.log(num_documents / tokens[token])

        # one vector per document
        for tokenized_doc in self.tokenized_docs:
            self.vectors.append(self._to_vector(tokenized_doc))

    def _to_vector(self, tokenized_doc):
        length = len(tokenized_doc)
        if length == 0:
            return {}

        vector = {}
        counts = Counter(tokenized_doc)
        for word, count in counts.items():
            if word in self.idf_values:  # skip words the index has never seen (queries)
                tf = count / length
                vector[word] = tf * self.idf_values[word]

        # normalize to length 1, so a dot product equals cosine similarity
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm == 0:
            return {}
        return {word: weight / norm for word, weight in vector.items()}

    def vectorize(self, text):
        # vector of any text (used for the query), with the documents' idf
        return self._to_vector(self.tokenize(text))

    def score(self, query):
        # cosine similarity between the query and every document, in document order
        query_vector = self.vectorize(query)
        scores = []
        for doc_vector in self.vectors:
            score = 0.0
            for word, weight in query_vector.items():
                score += weight * doc_vector.get(word, 0.0)
            scores.append(score)
        return scores

    def search(self, query, k=5):
        # (document position, score) of the k best documents
        scores = self.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked[:k]]