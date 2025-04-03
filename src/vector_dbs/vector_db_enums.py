from enum import Enum

class VectorDBEnums(Enum):
    FAISS = "FAISS"
    QDRANT = "QDRANT"

class DistanceMethodEnums(Enum):
    COSINE = "cosine"
    DOT = "dot"