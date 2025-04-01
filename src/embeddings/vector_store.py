from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStore:
    """Handles storing and retrieving document embeddings."""

    def __init__(self, model="text-embedding-ada-002"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model)
        self.vector_db = None

    def build_vector_store(self, documents):
        """Stores document embeddings in FAISS."""
        self.vector_db = FAISS.from_documents(documents, self.embeddings)

    def save_vector_store(self, path="faiss_index"):
        """Saves the FAISS index to disk."""
        self.vector_db.save_local(path)

    def load_vector_store(self, path="faiss_index"):
        """Loads a FAISS index from disk."""
        self.vector_db = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)

    def query(self, query_text, k=5):
        """Retrieves the top-k most relevant chunks."""
        return self.vector_db.similarity_search(query_text, k=k)
