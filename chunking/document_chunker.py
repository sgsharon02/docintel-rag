"""
Document Chunker for DocIntel

Creates overlapping chunks from page-level documents while
preserving metadata for traceability and verification.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
            ],
        )

    def chunk_documents(self, documents):

        chunked_docs = []

        for doc in documents:

            splits = self.splitter.split_text(doc.page_content)

            for i, chunk in enumerate(splits):

                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                    },
                )

                chunked_docs.append(chunk_doc)

        return chunked_docs