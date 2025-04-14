
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare():
    df = load_dataset()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df.to_pickle("data/train_df.pkl")
    test_df.to_pickle("data/test_df.pkl")
    print("Saved train/test splits to data/ folder")

if __name__ == "__main__":
    prepare()
