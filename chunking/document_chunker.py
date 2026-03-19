"""
Document Chunker for DocIntel
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
import re


ITEM_SECTION = re.compile(r"^item\s+\d+[a-zA-Z]?", re.IGNORECASE)
UPPER_SECTION = re.compile(r"^[A-Z][A-Z\s]{10,}$")
TITLE_SECTION = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}$")

def is_heading(line: str):
    line = line.strip()
    if not line:
        return False
    words = line.split()
    # Too short
    if len(line) < 12:
        return False
    # Too long (likely paragraph)
    if len(line) > 80:
        return False
    # Too many words (paragraph)
    if len(words) > 8:
        return False
    # Reject sentences
    if "." in line or "," in line or ":" in line:
        return False
    # Count capitalized words
    capitalized = sum(1 for w in words if w[0].isupper())
    # Heading usually has majority capitalized words
    if capitalized / len(words) < 0.6:
        return False
    return True 

class DocumentChunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )
    def chunk_documents(self, documents):
        chunked_docs = []
        current_section = "Unknown Section"
        for doc in documents:
            text = doc.page_content
            lines = text.split("\n")
            # detect section on this page
            for line in lines:
                line_clean = line.strip()
                if len(line_clean) < 12:
                    continue         
                if "." in line_clean:
                    continue
                if "," in line_clean:
                    continue
                if len(line_clean) > 80:
                    continue

                word_count = len(line_clean.split())
                # Item sections
                if ITEM_SECTION.match(line_clean):
                    current_section = line_clean
                    break
                # ALL CAPS headings
                if UPPER_SECTION.match(line_clean) and word_count >= 2:
                    current_section = line_clean
                    break
                # Title Case headings
                if TITLE_SECTION.match(line_clean):
                    current_section = line_clean
                    break
            splits = self.splitter.split_text(text)

            for i, chunk in enumerate(splits):
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "section": current_section,
                    },
                )
                chunked_docs.append(chunk_doc)
        return chunked_docs