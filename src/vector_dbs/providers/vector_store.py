from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


class VectorStore:
    """Handles storing and retrieving document embeddings in FAISS."""

    def __init__(self):
        self.vector_db = None

    def build_vector_store(self, documents, embeddings_function):
        """Stores document embeddings in FAISS using an external embedding function."""
        texts = [doc.page_content for doc in documents]  # Extract text content
        embeddings = [embeddings_function(text) for text in texts]  # Generate embeddings using the provided function
        embeddings = list(embeddings)

        # Ensure embeddings are in a list of lists format
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        text_embedding_pairs = zip(texts, embeddings)
        # Store in FAISS
        self.vector_db = FAISS.from_embeddings(text_embedding_pairs, documents)

    def save_vector_store(self, path="faiss_index"):
        """Saves the FAISS index to disk."""
        self.vector_db.save_local(path)

    def load_vector_store(self, path="faiss_index", embeddings_function=None):
        """Loads a FAISS index from disk."""
        """Retrieves the top-k most relevant chunks."""
        if embeddings_function is None:
            raise ValueError("An embedding function must be provided for load_vector_store.")
        self.vector_db = FAISS.load_local(path, embeddings_function,
                                          allow_dangerous_deserialization=True)

    def query(self, query_text: str, k=5):
        """Retrieves the top-k most relevant chunks."""

        results = self.vector_db.similarity_search_with_score(query_text, k=k)
        return results