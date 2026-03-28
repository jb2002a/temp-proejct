from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from llama_index.readers.file import PyMuPDFReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 벡터 스토리지 경로
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# 벡터 스토리지 컬렉션 이름
CHROMA_COLLECTION_NAME = "diabetes_clinical_guideline_collection"

# PDF 폴더 경로
PDF_FOLDER_PATH = PROJECT_ROOT / "pdfs" 

# 테스트 데이터셋 경로
TEST_DATASET_PATH = PROJECT_ROOT / "dataset" / "test_dataset" / "testdata.json"

# PDF Reader
PDF_READER = PyMuPDFReader()

# 청킹 로직
CHUKING_LOGIC = SentenceSplitter()

# 임베딩 모델 명
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# top-k 수치
SIMILARITY_TOP_K = 3

# 리포트 폴더 경로
REPORT_FOLDER_PATH = PROJECT_ROOT / "report"