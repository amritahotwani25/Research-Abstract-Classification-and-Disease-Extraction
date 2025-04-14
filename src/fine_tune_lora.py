
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import load_dataset
from src.train_model import tokenize_data, AbstractDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch

def fine_tune_lora():
    # Load dataset
    df = load_dataset()

    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenize_data(train_df, tokenizer)
    test_encodings = tokenize_data(test_df, tokenizer)

    train_dataset = AbstractDataset(train_encodings, list(train_df['label']))
    test_dataset = AbstractDataset(test_encodings, list(test_df['label']))

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "v_lin"]
    )
    lora_model = get_peft_model(base_model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    print("✅ LoRA fine-tuning complete.")

if __name__ == "__main__":
    fine_tune_lora()
