# BeautifulSoup is required to use the custom handlers
from bs4 import Tag
from langchain_text_splitters import HTMLSemanticPreservingSplitter

def read_html():
    with open("documents.html", "r", encoding="utf-8") as f:
        html_string = f.read()
    return html_string

def code_handler(element: Tag) -> str:
    data_lang = element.get("data-lang")
    code_format = f"<code:{data_lang}>{element.get_text()}</code>"

    return code_format

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLSemanticPreservingSplitter(
    headers_to_split_on=headers_to_split_on,
    separators=["\n\n", "\n", ". ", "! ", "? "],
    max_chunk_size=500,
    elements_to_preserve=["table", "ul", "ol", "code"],. 
    denylist_tags=["script", "style", "head"],
    custom_handlers={"code": code_handler},
)

if __name__ == "__main__":
    #python -m src.rag.pre_processing.split_test2
    html_string = read_html()
    documents = splitter.split_text(html_string)
    with open("html_header_splits2.txt", "w", encoding="utf-8") as f:
        for idx, document in enumerate(documents):
            f.write(f"[{idx}]\n")
            f.write("content: ")
            f.write(document.page_content)
            f.write("\n")
            f.write("metadata: ")
            f.write(", ".join([f"{k}: {v}" for k, v in document.metadata.items()]))
            f.write("\n\n")
