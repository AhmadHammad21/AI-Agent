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
        embeddings = embeddings_function(texts)  # Generate embeddings using the provided function
        
        # Ensure embeddings are in a list of lists format
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        text_embedding_pairs = zip(texts, embeddings)
        # Store in FAISS
        self.vector_db = FAISS.from_embeddings(text_embedding_pairs, documents)

    def save_vector_store(self, path="faiss_index"):
        """Saves the FAISS index to disk."""
        self.vector_db.save_local(path)

    def load_vector_store(self, path="faiss_index"):
        """Loads a FAISS index from disk."""
        self.vector_db = FAISS.load_local(path, allow_dangerous_deserialization=True)

    def query(self, query_text, k=5, embeddings_function=None):
        """Retrieves the top-k most relevant chunks."""
        if embeddings_function is None:
            raise ValueError("An embedding function must be provided for querying.")

        query_embedding = embeddings_function([query_text])  # Convert query to embedding
        return self.vector_db.similarity_search_by_vector(query_embedding[0], k=k)