
import os
import sys
import json
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load test data
test_df = pd.read_pickle("data/test_df.pkl").reset_index(drop=True)

# Load local LLM
llm = OllamaLLM(model="phi")
print("🧠 Ollama model loaded. Starting extraction...")

# Define strict JSON-only prompt
prompt = PromptTemplate.from_template(
    '''
You are a biomedical AI assistant. Given the research abstract below, return a JSON object with:
- "extracted_diseases": list of disease names mentioned
- "citations": list of references or in-text citations mentioned

⚠️ Respond with ONLY valid JSON. No explanations, no commentary.

Abstract:
{abstract}

JSON:
'''
)

# Chain: Prompt → LLM → Post-process
extraction_chain = prompt | llm

results = []

for idx, row in test_df.head(2).iterrows():
    try:
        print(f"\n🔍 Abstract {idx+1} — PubMed ID: {row['pubmed_id']}")
        # raw_output = extraction_chain.invoke({"abstract": row["abstract"]})
        raw_output = extraction_chain.invoke({"abstract": "Cancer is danger. HIV is more dangerous."})
        print("🧾 Raw LLM output:", repr(raw_output))

        json_start = raw_output.find("{")
        json_substring = raw_output[json_start:].strip()
        output = json.loads(json_substring)

        results.append({
            "abstract_id": row["pubmed_id"],
            "predicted_labels": ["Cancer"] if row["label"] == 1 else ["Non-Cancer"],
            "extracted_diseases": output.get("extracted_diseases", []),
            "citations": output.get("citations", [])
        })

    except Exception as e:
        print("❌ Failed to process abstract:", str(e))
        results.append({
            "abstract_id": row["pubmed_id"],
            "predicted_labels": ["Cancer"] if row["label"] == 1 else ["Non-Cancer"],
            "extracted_diseases": [],
            "citations": [],
            "error": str(e),
            "raw_output": raw_output if 'raw_output' in locals() else ""
        })

# Save to JSON
os.makedirs("results", exist_ok=True)
with open("results/velsera_submission_output.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Extraction complete. Results saved to results/velsera_submission_output.json")
