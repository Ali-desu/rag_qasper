from datasets import load_dataset

class Ingestor:
    def __init__(self):
        pass

    def load_qasper_validation(self):
        dataset = load_dataset("parquet",data_files="data/qasper_validation.parquet", split="train")
        return dataset

    def chunk_data(self, dataset):
        #first we divive the dataset by sections, then we chunk each section into smaller pieces
        #the metadata of each chunk will include the section title and the chunk index and paperid
        chunked_data = []

        for paper in dataset:
            paper_id = paper["id"]
            text = paper["full_text"]
            index = 0

            for section,paragraphs in zip(text["section_name"],text["paragraphs"]):
                for p in paragraphs:
                    #skip id paragraphs that are empty
                    if not p.strip():
                        continue    
                    #each chunk is the whole paragraph
                    chunked_data.append({
                        "id": f"{paper_id}::{index}",
                        "paper_id": paper_id,
                        "section": section,
                        "chunk_index": index,
                        "text": p
                    })
                    index += 1
                        

        return chunked_data
                    

    def ingest(self):
        # Logic to ingest data from the source
        pass