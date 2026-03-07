import os
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from fastapi import FastAPI
from api.routes import query, ingestion, health, index, ingestion_status
import logging
import sys
from dotenv import load_dotenv
load_dotenv()
# ...existing imports...

# Configure logging before FastAPI app initialization
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/api.log"),
    ]
)

# Suppress noisy third-party loggers
logging.getLogger("rapidocr").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

app = FastAPI(title="DocIntel API")

app.include_router(query.router)
app.include_router(ingestion.router)
app.include_router(health.router)
app.include_router(ingestion_status.router)
app.include_router(index.router)
