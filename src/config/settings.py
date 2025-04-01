# settings.py
PDF_DIRECTORY = "data/docs"
PROCESSED_DIRECTORY = "data/processed"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "ALLaM-AI/ALLaM-7B-Instruct-preview"
VECTOR_STORE_PATH = "data/faiss_index"
TOP_K = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200