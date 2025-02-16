import re
import pandas as pd
import pymupdf
import unicodedata

def extract_text(file_path):
    data = []
    try:
        with pymupdf.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                images = page.get_images(full=True)
                annotations = list(page.annots()) if page.annots() else []
                drawings = page.get_drawings()
                widgets = list(page.widgets()) if page.annots() else []

                data.append({"Page": page_num, "Text": page_text if page_text else "[No readable text]",
                             "Images": images if images else [], "Annotations": annotations if annotations else [],
                             "Drawings": drawings if drawings else [], "Widgets": widgets if widgets else []})

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading PDF: {e}")

    return pd.DataFrame(data)

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode text
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)  # Remove non-printable ASCII characters (excluding spaces)
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def get_clean_dataframe(file_path):
    df = extract_text(file_path)
    df["Text"] = df["Text"].apply(clean_text)
    return df

