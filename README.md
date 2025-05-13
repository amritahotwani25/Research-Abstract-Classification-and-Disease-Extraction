
# Research Abstract Classification & Disease Extraction Pipeline

## 🧠 Overview
This pipeline classifies PubMed abstracts into **Cancer** vs **Non-Cancer**, provides confidence scores, and extracts disease mentions + citations using a fine-tuned model and an LLM.

---

## ✅ Features
- Fine-tuning with LoRA using DistilBERT
- Structured JSON output per abstract
- Classification with confidence scores
- Disease & citation extraction using Ollama LLM (`phi`)

---

## 🚀 How to Run (Unified)
```bash
PYTHONPATH=. python src/predict_and_extract.py
```

This will:
- Load your fine-tuned classification model
- Predict the label + confidence
- Extract diseases and citations using `phi` via Ollama
- Save results to:
```
results/final_predictions.json
```

---

## 📦 Output Format
```json
{
  "abstract_id": "31065274",
  "predicted_label": "Cancer",
  "confidence_scores": {
    "Cancer": 0.92,
    "Non-Cancer": 0.08
  },
  "extracted_diseases": ["Breast Cancer"],
  "citations": ["Smith et al., 2021"]
}
```

---

## 📁 Project Structure
```
src/
  ├── load_data.py
  ├── train_model.py
  ├── fine_tune_lora.py
  ├── compare_models.py
  └── predict_and_extract.py ✅
```

---

### 🔍 Model Selection & Justification

We selected **DistilBERT (distilbert-base-uncased)** as our baseline classification model for the following reasons:

- ✅ **Efficient architecture**: DistilBERT retains ~97% of BERT’s performance while being 40% smaller and faster, making it ideal for fast experimentation and deployment.
- ✅ **Pre-trained on general English text**: Suitable for biomedical abstracts, which are generally well-structured formal text.
- ✅ **Strong performance out-of-the-box**: The model achieved ~98% classification accuracy on the cancer vs. non-cancer task, outperforming even the LoRA fine-tuned version in our case.
- ✅ **Well-supported ecosystem**: Easily integrated with Hugging Face Transformers, compatible with `Trainer`, and extendable with `peft` for LoRA if needed.

Thus, we used `distilbert-base-uncased` both as a baseline and for final predictions in the unified pipeline.


---

### ⚙️ Full Pipeline Steps (Optional)

If you'd like to train from scratch instead of using the pretrained model:

```bash
# 1. Prepare and clean the dataset from raw text files
PYTHONPATH=. python src/prepare_data.py

# 2. Train the baseline model (DistilBERT)
PYTHONPATH=. python src/train_model.py

# 3. (Optional) Fine-tune the baseline using LoRA
PYTHONPATH=. python src/fine_tune_lora.py

# 4. Compare model performance (baseline vs fine-tuned)
PYTHONPATH=. python src/compare_models.py

# 5. Run unified prediction + disease & citation extraction
PYTHONPATH=. python src/predict_and_extract.py
```

> 💡 Steps 1–4 are optional if you're using the already trained model saved under `models/baseline_model`.

The final output will be stored at:
```
results/final_predictions.json
```
