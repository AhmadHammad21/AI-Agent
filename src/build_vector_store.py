from loaders.pdf_loader import PDFLoader
from processors.text_processor import TextProcessor
from embeddings.vector_store import VectorStore
from utils.logger import get_logger
from utils.exceptions import PDFLoadError
from config.settings import (
    PDF_DIRECTORY, EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP
)
from dotenv import load_dotenv
load_dotenv()

# Set up logger
logger = get_logger(__name__)
    
def build_and_save_vector_embeddings():
    logger.info("Loading PDFs...")
    pdf_loader = PDFLoader(PDF_DIRECTORY)
    pdf_docs = pdf_loader.load_pdfs()
    if not pdf_docs:
        raise PDFLoadError("No PDFs found in the directory.")

    logger.info("Splitting text into chunks...")
    text_processor = TextProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
    documents = text_processor.split_documents(pdf_docs)

    logger.info("Storing embeddings...")
    vector_store = VectorStore(EMBEDDING_MODEL_NAME)
    vector_store.build_vector_store(documents)
    vector_store.save_vector_store(VECTOR_STORE_PATH)
    logger.info("Saved New embeddings...")


if __name__ == "__main__":
    build_and_save_vector_embeddings()