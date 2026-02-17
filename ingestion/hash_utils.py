import hashlib

def compute_docs_hash(documents):
    hasher = hashlib.sha256()

    sorted_docs = sorted(
        documents,
        key=lambda d: (
            d.metadata.get("source", ""),
            d.metadata.get("page", 0),
            d.page_content[:50],
        ),
    )

    for doc in sorted_docs:
        hasher.update(doc.page_content.encode("utf-8"))

    return hasher.hexdigest()
