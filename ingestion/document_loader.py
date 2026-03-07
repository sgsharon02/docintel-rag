import logging
import os
from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self):
        pass

    def load_pdf(self, file_path: str):
        logger.info(f"Loading PDF: {file_path}")
        reader = PdfReader(file_path)
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text.strip():
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "page": page_number + 1,
                },
            )

            documents.append(doc)

        logger.info(f"Loaded {len(documents)} pages from {os.path.basename(file_path)}")
        return documents

    def load_directory(self, folder_path: str):
        logger.info(f"Loading PDFs from directory: {folder_path}")
        all_docs = []
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for idx, file in enumerate(pdf_files, 1):
            path = os.path.join(folder_path, file)
            logger.info(f"[{idx}/{len(pdf_files)}] Processing {file}...")
            docs = self.load_pdf(path)
            all_docs.extend(docs)

        logger.info(f"Total documents loaded: {len(all_docs)}")
        return all_docs