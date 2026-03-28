import re

# C0 controls except \t \n \r
C0_CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# PDF에서 자주 섞이는 이상 글리프(Replacement char + PUA)
PDF_GLYPH_RE = re.compile(r"[\uFFFD\uE000-\uF8FF]")

def clean_pdf_text(text: str) -> str:
    text = C0_CONTROL_RE.sub("", text)
    text = PDF_GLYPH_RE.sub("", text)
    return text