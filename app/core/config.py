import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Dedup safety: whether the importer may ask an LLM to INVENT missing content fields
# (description/scope/objectives/expected_result). Default OFF — a duplicate-detection system
# must not compare on fabricated text. When off, missing fields stay empty and the row is
# flagged needs_review. See app/services/llm_normalizer_service.py.
ALLOW_LLM_CONTENT_FILL = os.getenv("ALLOW_LLM_CONTENT_FILL", "false").lower() == "true"

# SEDO / similarity settings.
# Default (False) matches the paper: an unrecognized concept is dropped and the
# structural/domain score falls (paper Sect. 5.5). Set True to fall back to token
# Jaccard when the ontology recognizes nothing on a dimension.
SEDO_FALLBACK_TOKENS = os.getenv("SEDO_FALLBACK_TOKENS", "false").lower() == "true"
# wpath k parameter (Zhu-Iglesias). Tune to match the paper's setup.
WPATH_K = float(os.getenv("WPATH_K", "0.8"))
