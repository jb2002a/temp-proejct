from src.rag.document_loader import load_documents
from ragas.utils import num_tokens_from_string

def main():
    docs = load_documents()
    tokens = [num_tokens_from_string(d.text or "") for d in docs]
    print("docs:", len(tokens))
    print("<=100:", sum(t <= 100 for t in tokens))
    print("101~500:", sum(101 <= t <= 500 for t in tokens))
    print(">500:", sum(t > 500 for t in tokens))

if __name__ == "__main__":
    # python -m src.test
    main()