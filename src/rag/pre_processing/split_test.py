from langchain_text_splitters import HTMLSemanticPreservingSplitter

#documents.html 읽기
def read_html():
    with open("documents.html", "r", encoding="utf-8") as f:
        html_string = f.read()
    return html_string


def split_html():
    html_string = read_html()
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
    ]
    html_splitter = HTMLSemanticPreservingSplitter(headers_to_split_on)
    html_header_splits = html_splitter.split_text(html_string)
    return html_header_splits

if __name__ == "__main__":
    #python -m src.rag.pre_processing.split_test
    html_header_splits = split_html()

    #저장
    with open("html_header_splits.txt", "w", encoding="utf-8") as f:
        for idx, split in enumerate(html_header_splits):
            f.write(f"[{idx}]\n")
            f.write("content: ")
            f.write(split.page_content)
            f.write("\n")
            f.write("metadata: ")
            f.write(", ".join([f"{k}: {v}" for k, v in split.metadata.items()]))
            f.write("\n\n")
