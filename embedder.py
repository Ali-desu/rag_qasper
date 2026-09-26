from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="Snowflake/snowflake-arctic-embed-s", batch_size=32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def embed_query(self, query):
        return self.model.encode(
            query,
            prompt_name="query",         # adds the model's own query prefix
            normalize_embeddings=True,
        )