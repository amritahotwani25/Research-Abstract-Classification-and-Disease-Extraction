
import os
import re
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
CANCER_PATH = os.getenv("CANCER_PATH")
NON_CANCER_PATH = os.getenv("NON_CANCER_PATH")

def clean_text(text):
    text = re.sub(r'\(.*?, \d{4}\)', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            text = f.read()

    pubmed_id = re.search(r'<ID:(\d+)>', text)
    title = re.search(r'Title:\s*(.*?)\n', text)
    abstract = re.search(r'Abstract:\s*(.*)', text, re.DOTALL)

    return {
        "pubmed_id": pubmed_id.group(1) if pubmed_id else None,
        "title": clean_text(title.group(1)) if title else "",
        "abstract": clean_text(abstract.group(1)) if abstract else ""
    }

def load_dataset():
    records = []
    for file in os.listdir(CANCER_PATH):
        parsed = parse_file(os.path.join(CANCER_PATH, file))
        if parsed["abstract"]:
            parsed["label"] = 1
            records.append(parsed)
    for file in os.listdir(NON_CANCER_PATH):
        parsed = parse_file(os.path.join(NON_CANCER_PATH, file))
        if parsed["abstract"]:
            parsed["label"] = 0
            records.append(parsed)
    return pd.DataFrame(records)
