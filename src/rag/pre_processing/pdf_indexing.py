import fitz
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent

#인덱싱용, 필요시 사용
def split_pdf(input_file, batch_size):
    # Open input_pdf
    input_pdf = fitz.open(input_file)
    num_pages = len(input_pdf)
    print(f"Total number of pages: {num_pages}")
 
    # Split input_pdf
    for start_page in range(0, num_pages, batch_size):
        end_page = min(start_page + batch_size, num_pages) - 1
 
        # Write output_pdf to file
        input_file_basename = os.path.splitext(input_file)[0]
        output_file = f"{input_file_basename}_{start_page}_{end_page}.pdf"
        print(output_file)
        with fitz.open() as output_pdf:
            output_pdf.insert_pdf(input_pdf, from_page=start_page, to_page=end_page)
            output_pdf.save(output_file)
 
    # Close input_pdf
    input_pdf.close()

if __name__ == "__main__":
    # python -m src.rag.pre_processing.pdf_indexing

    split_pdf(PROJECT_ROOT / "pdfs" / "강의자료(1~4장).pdf", 100)