from langchain_core.documents import Document
from pypdf import PdfReader
import os


class DocumentLoader:
    def __init__(self):
        pass

    def load_pdf(self, file_path: str):
        reader = PdfReader(file_path)
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_number + 1,
                },
            )

            documents.append(doc)

        return documents

    def load_directory(self, folder_path: str):
        all_docs = []

        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                path = os.path.join(folder_path, file)
                docs = self.load_pdf(path)
                all_docs.extend(docs)

        return all_docs
