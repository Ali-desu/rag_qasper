from Ingestor import Ingestor
from embedder import Embedder
from tfidf_index import TfidfIndex
from vector_db import VectorStore

def main():

    print("#begin initialisation of objects")
    ingestor = Ingestor()
    embedder = Embedder()
    vectorstore = VectorStore(dimension=embedder.dimension)

    ingestor.ingest(embedder,vectorstore,10)
    


if __name__ == "__main__":
    main()