from langchain_upstage import UpstageDocumentParseLoader
from dotenv import load_dotenv
from src.rag.common.config import PDF_FILE_PATH, JSONL_FILE_PATH
from langsmith import traceable
import json
from pathlib import Path
 
# 1회 요청 가능 페이지가 1천페이지이므로 분할하여 개발자가 요청
# 후에 필요시 병렬 처리 로직 추가 가능

# 직접 개발자가 pdf 쪼개고 개별적으로 UpstageApi 호출후에 element 별로 받은 뒤에 jsonl 파일로 저장하는 로직
load_dotenv(override=True)

@traceable
def load_documents():
    loader = UpstageDocumentParseLoader(PDF_FILE_PATH, output_format="markdown", split="element")
    documents = loader.load()
    return documents

@traceable
def save_documents_jsonl(
    documents,
    output_path: str = JSONL_FILE_PATH,
    source_file: str = PDF_FILE_PATH,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    current_subject = "na"
    NOISE_SINGLE_TOKENS = {"↘", "V"}
    # heading1이 화살표만 올 때는 이전 주제 유지 (저장되는 current_subject가 "# ↘"로 오염되는 것 방지)
    NOISE_HEADING_SUBJECTS = {"# ↘", "#↘"}

    with open(output_path, "a", encoding="utf-8") as f:
        for idx, doc in enumerate(documents):

            cat = (doc.metadata.get("category") or "").lower()
            text = (doc.page_content or "").strip()

            if cat == "heading1" and text:
                if text in NOISE_HEADING_SUBJECTS:
                    continue
                current_subject = text
                continue

            if text in NOISE_SINGLE_TOKENS:
                continue

            if cat in {"header", "footer"}:
                continue
            
            if not text:
                continue

            row = {
                "page_content": text,
                "category": doc.metadata.get('category', 'na'),
                "coordinates": doc.metadata.get('coordinates', 'na'),
                "id": doc.metadata.get('id', 'na'),
                "page": doc.metadata.get('page', 'na'),
                "current_subject" : current_subject,
                "source_file": source_file,
            }
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def forward_fill_subject_placeholder(
    jsonl_path: str,
    placeholder: str = "# ↘",
    default: str = "na",
) -> None:
    """JSONL에서 current_subject가 placeholder인 행만, 동일 source_file 내 직전 정상 주제로 채움."""
    path = Path(jsonl_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    last_good_by_source: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        src = row.get("source_file") or ""
        subj = (row.get("current_subject") or "").strip()
        if subj == placeholder:
            row["current_subject"] = last_good_by_source.get(src, default)
        else:
            last_good_by_source[src] = subj
        out.append(json.dumps(row, ensure_ascii=False, default=str))
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


@traceable
def run_pipeline_load_and_save():
    documents = load_documents()
    save_documents_jsonl(documents)

if __name__ == "__main__":
    # python -m src.rag.pre_processing.pdf_to_jsonl
    run_pipeline_load_and_save()


