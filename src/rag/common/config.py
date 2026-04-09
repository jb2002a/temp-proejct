
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

MODEL = "gpt-4o-mini"
EMBEDING_MODEL = "BAAI/bge-m3"
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "guideline_collection"
PDF_FILE_PATH = PROJECT_ROOT / "pdfs" / "당뇨병 임상진료지침-15-148.pdf"
