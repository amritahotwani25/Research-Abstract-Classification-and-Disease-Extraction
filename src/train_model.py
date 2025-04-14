
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

class AbstractDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_data(df, tokenizer, max_length=512):
    return tokenizer(
        list(df['abstract']),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def train_baseline():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_df = pd.read_pickle("data/train_df.pkl")
    test_df = pd.read_pickle("data/test_df.pkl")

    train_enc = tokenize_data(train_df, tokenizer)
    test_enc = tokenize_data(test_df, tokenizer)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = AbstractDataset(train_enc, list(train_df['label']))
    test_dataset = AbstractDataset(test_enc, list(test_df['label']))

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="models/baseline_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)
    trainer.train()
    model.save_pretrained("models/baseline_model")
    tokenizer.save_pretrained("models/baseline_model")
    print("✅ Baseline model saved to models/baseline_model")

if __name__ == "__main__":
    train_baseline()
