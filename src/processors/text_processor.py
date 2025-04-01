from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """Processes text by splitting it into smaller chunks for retrieval."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, documents):
        """Splits documents into smaller chunks for better retrieval."""
        return self.text_splitter.split_documents(documents)