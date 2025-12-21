# app/normalizer.py
import re

def normalize_arabic(text):
    # Remove diacritics
    text = re.sub(r'[\u064B-\u0652]', '', text)

    # Remove tatweel
    text = text.replace("ـ", "")

    # Normalize spaces around "و"
    text = re.sub(r'\s*و\s*', ' و', text)

    # Remove double spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
