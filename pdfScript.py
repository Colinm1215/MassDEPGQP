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
    """
    Initializes the SymSpell spell checker with an English frequency dictionary.

    Returns:
        SymSpell: Configured spell correction instance
    """
    sym_spell = SymSpell()
    dictionary_path = os.path.join(os.path.dirname(__file__), "frequency_dictionary_en_82_765.txt")
    if os.path.exists(dictionary_path):
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

spell_checker = initialize_spell_checker()
spelling_cache = {}

def should_correct_token(token):
    """
    Determine if a word token should be spell-checked.

    Args:
        token: A word candidate

    Returns:
        bool: Whether the token is eligible for correction
    """
    if any(ch.isdigit() for ch in token):
        return False
    if token.isupper() and len(token) > 1:
        return False
    if len(token) <= 2:
        return False
    return True

def correct_spelling(text, max_edit_distance=2):
    """
    Applies spell correction to a string using a cached SymSpell instance.

    Args:
        text: Raw string to correct
        max_edit_distance: Maximum Levenshtein distance for suggestions

    Returns:
        str: Spell-corrected version of the text
    """
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
    """
    Extracts text content from each page in a PDF using PyMuPDF.

    Args:
        file_path: Full path to the PDF file

    Returns:
        pd.DataFrame: One row per page with 'Page' and 'Text' columns
    """
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
    """
    Cleans extracted text by:
    - Normalizing unicode to NFKC
    - Removing non-ASCII characters
    - Collapsing excessive whitespace

    Args:
        text: Raw page text

    Returns:
        str: Cleaned text
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)  # Remove non-printable characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_clean_dataframe(file_name, file_path):
    """
    Orchestrates full PDF-to-DataFrame preprocessing:
    - Extracts raw text from each page
    - Normalizes and cleans the text
    - Applies spell correction with caching
    - Adds filename metadata

    Args:
        file_name: Uploaded file name
        file_path: Full path to the uploaded file

    Returns:
        pd.DataFrame: Cleaned and corrected dataframe with:
                      ['Page', 'Text', 'filename', 'Corrected_Text']
    """
    df = extract_text(file_path)
    df["filename"] = file_name.split(".")[0]
    df["Text"] = df["Text"].apply(clean_text)
    df["Corrected_Text"] = df["Text"].apply(correct_spelling)
    return df
