class PDFLoadError(Exception):
    """Exception raised when a PDF file cannot be loaded."""
    pass

class EmbeddingError(Exception):
    """Exception raised when embedding storage fails."""
    pass

class QueryError(Exception):
    """Exception raised when querying the vector store fails."""
    pass
