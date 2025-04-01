import os
from langchain_community.document_loaders import PyPDFLoader

class PDFLoader:
    """Loads PDF documents from a given directory."""
    def __init__(self, directory: str):
        self.directory = directory
        self.pdf_docs = []

    def load_pdfs(self):
        """Loads all PDF files from the specified directory."""
        for file in os.listdir(self.directory):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.directory, file))
                self.pdf_docs.extend(loader.load())
        return self.pdf_docs

