import faiss
import os
import pickle
from typing import List, Tuple
from ..vector_db_interface import VectorDBInterface
from ..vector_db_enums import DistanceMethodEnums


class FaissDBProvider(VectorDBInterface):
    """FAISS-based vector database that assumes embeddings are generated externally."""
    ## NOTE: NOT USED
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.index = None
        self.metadata_store = {}  # To store metadata alongside vectors

    def connect(self):
        """Loads an existing FAISS index if available."""
        if os.path.exists(self.db_path):
            self.load_vector_store()
        else:
            raise FileNotFoundError(f"FAISS index not found at {self.db_path}")

    def disconnect(self):
        """Clears the FAISS index from memory."""
        self.index = None
        self.metadata_store = {}

    def is_collection_existed(self, collection_name: str) -> bool:
        """FAISS does not support multiple collections, so always return True if index exists."""
        return self.index is not None

    def list_all_collections(self) -> List:
        """FAISS does not support named collections, return a placeholder."""
        return ["default_collection"]

    def get_collection_info(self, collection_name: str) -> dict:
        """Returns metadata about the FAISS index."""
        if self.index is None:
            raise ValueError("No FAISS index loaded.")
        return {"index_size": self.index.ntotal}

    def delete_collection(self, collection_name: str):
        """Deletes the FAISS index."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.index = None
        self.metadata_store = {}

    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """Creates a new FAISS index."""
        if do_reset:
            self.delete_collection(collection_name)

        self.index = faiss.IndexFlatL2(embedding_size)
        self.metadata_store = {}

    def insert_one(self, collection_name: str, vector: list, metadata: dict = None, record_id: str = None):
        """Inserts a single vector into the FAISS index."""
        if self.index is None:
            raise ValueError("FAISS index has not been created.")

        vector = faiss.vector_to_array(vector).reshape(1, -1)  # Ensure correct shape
        self.index.add(vector)
        
        record_id = record_id or str(self.index.ntotal - 1)
        self.metadata_store[record_id] = metadata

    def insert_many(self, collection_name: str, vectors: List[list], metadata: List[dict] = None, record_ids: List[str] = None):
        """Inserts multiple vectors into the FAISS index."""
        if self.index is None:
            raise ValueError("FAISS index has not been created.")

        vectors = faiss.vector_to_array(vectors).reshape(len(vectors), -1)
        self.index.add(vectors)

        if metadata is None:
            metadata = [{}] * len(vectors)

        if record_ids is None:
            record_ids = [str(i) for i in range(self.index.ntotal - len(vectors), self.index.ntotal)]

        for rid, meta in zip(record_ids, metadata):
            self.metadata_store[rid] = meta

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5) -> List[Tuple[int, float, dict]]:
        """Searches the FAISS index for the most similar vectors."""
        if self.index is None:
            raise ValueError("FAISS index has not been created.")

        vector = faiss.vector_to_array(vector).reshape(1, -1)
        distances, indices = self.index.search(vector, limit)

        results = []
        for i in range(len(indices[0])):
            record_id = str(indices[0][i])
            metadata = self.metadata_store.get(record_id, {})
            results.append((indices[0][i], distances[0][i], metadata))

        return results

    def save_vector_store(self):
        """Saves the FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No FAISS index to save.")

        faiss.write_index(self.index, self.db_path)
        with open(self.db_path + "_metadata.pkl", "wb") as f:
            pickle.dump(self.metadata_store, f)

    def load_vector_store(self):
        """Loads the FAISS index and metadata from disk."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("FAISS index file not found.")

        self.index = faiss.read_index(self.db_path)

        metadata_path = self.db_path + "_metadata.pkl"
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                self.metadata_store = pickle.load(f)
        else:
            self.metadata_store = {}