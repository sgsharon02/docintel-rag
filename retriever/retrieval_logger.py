import os
from datetime import datetime


LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "retrieval.log")


def log_retrieval(query, documents, verification_status=None):
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n============================\n")
        f.write(f"TIME: {datetime.now()}\n")
        f.write(f"QUERY: {query}\n")
        f.write(f"RETRIEVED_CHUNKS: {len(documents)}\n")

        for doc in documents:
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            f.write(f"  - {src} (page {page})\n")

        if verification_status:
            f.write(f"VERIFICATION: {verification_status}\n")
