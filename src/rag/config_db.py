from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME = "diabetes_clinical_guideline_collection"
