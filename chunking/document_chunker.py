"""
Document Chunker for DocIntel

Creates overlapping chunks from page-level documents while
preserving metadata for traceability and verification.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_documents(self, documents):
        """
        Split page-level documents into smaller chunks.
        Preserve page metadata.
        """
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
