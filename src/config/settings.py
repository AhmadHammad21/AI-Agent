from dataclasses import dataclass

@dataclass
class AppSettings:
    PDF_DIRECTORY: str = "../data/docs"
    PROCESSED_DIRECTORY: str = "data/processed"

    # ================================ LLM Settings ================================
    # GENERATION_MODEL_ID: str = "gpt-4o-mini"
    GENERATION_MODEL_ID: str = "ALLaM-AI/ALLaM-7B-Instruct-preview"
    # EMBEDDING_MODEL_ID: str = "text-embedding-3-small"#sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL_ID: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL_SIZE: int = 384

    GENERATION_BACKEND: str = "HUGGINGFACE"#"OPENAI"
    EMBEDDING_BACKEND: str = "HUGGINGFACE"#OPENAI"

    INPUT_DAFAULT_MAX_CHARACTERS: int = 1024
    GENERATION_DAFAULT_MAX_TOKENS: int = 1024
    GENERATION_DAFAULT_TEMPERATURE: float = 0.1
    TOP_K: int = 10
    TOP_P: float = 0.95

    # ================================ Vector Database Settings ================================
    VECTOR_STORE_PATH: str = "../data/faiss_index" # ../ because we go into src directory 
    TOP_SIMILARITY_K: int = 3
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200



    # FILE_ALLOWED_TYPES=["text/plain", "application/pdf"]
    # FILE_MAX_SIZE=10
    # FILE_DEFAULT_CHUNK_SIZE=512000 # 512KB

    # MONGODB_URL="mongodb://admin:admin@localhost:27007"
    # MONGODB_DATABASE="mini-rag"



settings = AppSettings()
