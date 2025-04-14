
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class AbstractDataset(torch.utils.data.Dataset):
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

def evaluate():
    tokenizer = AutoTokenizer.from_pretrained("models/baseline_model")
    test_df = pd.read_pickle("data/test_df.pkl").reset_index(drop=True)
    test_enc = tokenize_data(test_df, tokenizer)
    test_dataset = AbstractDataset(test_enc, list(test_df['label']))

    print("\n✅ Evaluating Baseline Model:")
    baseline_model = AutoModelForSequenceClassification.from_pretrained("models/baseline_model")
    baseline_trainer = Trainer(model=baseline_model)
    baseline_preds = baseline_trainer.predict(test_dataset).predictions.argmax(axis=-1)

    print("Accuracy:", accuracy_score(test_df['label'], baseline_preds))
    print("F1 Score:", f1_score(test_df['label'], baseline_preds))
    print("Confusion Matrix:\n", confusion_matrix(test_df['label'], baseline_preds))

    print("\n✅ Evaluating LoRA-Tuned Model (with adapter):")
    lora_model = PeftModel.from_pretrained(
        AutoModelForSequenceClassification.from_pretrained("models/baseline_model"),
        "models/lora_model"
    )
    lora_trainer = Trainer(model=lora_model)
    lora_preds = lora_trainer.predict(test_dataset).predictions.argmax(axis=-1)

    print("Accuracy:", accuracy_score(test_df['label'], lora_preds))
    print("F1 Score:", f1_score(test_df['label'], lora_preds))
    print("Confusion Matrix:\n", confusion_matrix(test_df['label'], lora_preds))

if __name__ == "__main__":
    evaluate()
