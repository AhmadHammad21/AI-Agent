from loaders.pdf_loader import PDFLoader
from processors.text_processor import TextProcessor
from vector_dbs.providers.vector_store import VectorStore
from llms.llm_provider_factory import LLMProviderFactory
from utils.logger import get_logger
from utils.exceptions import PDFLoadError
from config.settings import settings
from config.config import config
from dotenv import load_dotenv
load_dotenv()


logger = get_logger(__name__)


def build_and_save_vector_embeddings():
    logger.info("Loading PDFs...")
    pdf_loader = PDFLoader(settings.PDF_DIRECTORY)
    pdf_docs = pdf_loader.load_pdfs()
    if not pdf_docs:
        raise PDFLoadError("No PDFs found in the directory.")

    llm_provider_factory = LLMProviderFactory(config, settings)
    embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                         embedding_size=settings.EMBEDDING_MODEL_SIZE)
    
    logger.info("Splitting text into chunks...")
    text_processor = TextProcessor(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    documents = text_processor.split_documents(pdf_docs)

    logger.info("Storing embeddings...")
    vector_store = VectorStore()
    vector_store.build_vector_store(documents, embedding_client.embed_text)
    vector_store.save_vector_store(settings.VECTOR_STORE_PATH)

    logger.info(f"Saved New embeddings... to {settings.VECTOR_STORE_PATH}")


if __name__ == "__main__":
    build_and_save_vector_embeddings()