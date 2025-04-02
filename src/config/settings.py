from dataclasses import dataclass

@dataclass
class AppSettings:
    PDF_DIRECTORY: str = "../data/docs"
    PROCESSED_DIRECTORY: str = "data/processed"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL_NAME: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    VECTOR_STORE_PATH: str = "../data/faiss_index" # ../ because we go into src directory 
    TOP_K: int = 3
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200


settings = AppSettings()
