
import os
import sys
import json
import pandas as pd
import torch
import re
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load fine-tuned model + tokenizer
model_path = "models/baseline_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

# Load local LLM
llm = OllamaLLM(model="phi")
print("Ollama model loaded. Starting extraction...")

# Define strict JSON-only prompt
prompt = PromptTemplate.from_template(
    '''
You are a biomedical AI assistant. Given the research abstract below, return a JSON object with:

{{
  "extracted_diseases": [list of disease names],
  "citations": [list of citation strings]
}}

Return only the JSON object, without any commentary or explanation.
Do not say “Sure” or “Here is the JSON”.

Abstract:
{abstract}

JSON:
'''
)
extraction_chain = prompt | llm

# Load test set
test_df = pd.read_pickle("data/test_df.pkl").reset_index(drop=True)
print(test_df.head(2))

results = []

print("🚀 Running model predictions and extractions...")

for idx, row in test_df.iterrows():
    print(row['abstract'])
    try:
        abstract = row["abstract"]
        inputs = tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=1).numpy()[0]
            predicted_class = probs.argmax()
            predicted_label = "Cancer" if predicted_class == 1 else "Non-Cancer"

        print(f"\n🔍 Abstract {idx+1} — PubMed ID: {row['pubmed_id']}")
        raw_output = extraction_chain.invoke({"abstract": abstract})
        print("🧾 Raw LLM output:", repr(raw_output))

        # Clean + parse JSON output from LLM               
        json_start = raw_output.find("{")
        json_end = raw_output.find("}", json_start) + 1  # capture one full JSON object
        json_candidate = raw_output[json_start:json_end]

        # Optional: Handle improperly escaped backslashes or quotes
        cleaned_output = json_candidate.strip()

        parsed = json.loads(cleaned_output)


        results.append({
            "abstract_id": row["pubmed_id"],
            "predicted_label": predicted_label,
            "confidence_scores": {
                "Cancer": round(float(probs[1]), 4),
                "Non-Cancer": round(float(probs[0]), 4)
            },
            "extracted_diseases": parsed.get("extracted_diseases", []),
            "citations": parsed.get("citations", [])
        })

    except Exception as e:
        print("❌ Failed to process abstract:", e)
        results.append({
            "abstract_id": row["pubmed_id"],
            "predicted_label": None,
            "confidence_scores": {},
            "extracted_diseases": [],
            "citations": [],
            "error": str(e)
        })

# Save final structured output
os.makedirs("results", exist_ok=True)
with open("results/final_predictions.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Extraction complete. Results saved to results/final_predictions.json")
