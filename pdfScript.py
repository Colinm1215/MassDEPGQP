import io
import re
import pandas as pd
import pymupdf
import unicodedata
from PIL import Image
import symspellpy
from symspellpy import SymSpell
import os

def initialize_spell_checker():
    sym_spell = SymSpell()
    dictionary_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
    if os.path.exists(dictionary_path):
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

spell_checker = initialize_spell_checker()
spelling_cache = {}

def should_correct_token(token):
    if any(ch.isdigit() for ch in token):
        return False
    if token.isupper() and len(token) > 1:
        return False
    if len(token) <= 2:
        return False
    return True

def correct_spelling(text, max_edit_distance=2):
    global spelling_cache

    words = text.split()
    corrected_words = []

    for w in words:
        if w not in spelling_cache:
            if should_correct_token(w):
                suggestions = spell_checker.lookup(w, symspellpy.Verbosity.CLOSEST, max_edit_distance=max_edit_distance)
                if suggestions:
                    spelling_cache[w] = suggestions[0].term
                else:
                    spelling_cache[w] = w
            else:
                spelling_cache[w] = w
        corrected_words.append(spelling_cache[w])

    return " ".join(corrected_words)

def extract_text(file_path):
    data = []
    try:
        with pymupdf.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text("text").strip()
                data.append({"Page": page_num, "Text": page_text if page_text else "[No readable text]"})
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return pd.DataFrame(data)

def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)  # Remove non-printable characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_clean_dataframe(file_name, file_path):
    df = extract_text(file_path)
    df["filename"] = file_name.split(".")[0]
    df["Text"] = df["Text"].apply(clean_text)
    df["Corrected_Text"] = df["Text"].apply(correct_spelling)
    return df
