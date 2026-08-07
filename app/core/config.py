import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# SEDO / similarity settings.
# Default (False) matches the paper: an unrecognized concept is dropped and the
# structural/domain score falls (paper Sect. 5.5). Set True to fall back to token
# Jaccard when the ontology recognizes nothing on a dimension.
SEDO_FALLBACK_TOKENS = os.getenv("SEDO_FALLBACK_TOKENS", "false").lower() == "true"
# wpath k parameter (Zhu-Iglesias). Tune to match the paper's setup.
WPATH_K = float(os.getenv("WPATH_K", "0.8"))

