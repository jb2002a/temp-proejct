from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 벡터 스토리지 경로
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# 벡터 스토리지 컬렉션 이름
CHROMA_COLLECTION_NAME = "diabetes_clinical_guideline_collection"

# PDF 폴더 경로
pdf_folder_path = PROJECT_ROOT / "pdfs" 

# 청킹 로직
splitter = SentenceSplitter()

# 임베딩 모델 명
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"