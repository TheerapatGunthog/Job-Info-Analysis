import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline
from collections import Counter
import json

# Load the model
model_name = "GalalEwida/LLM-BERT-Model-Based-Skills-Extraction-from-jobdescription"
ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)

INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
RAW_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/raw")
df = pd.read_csv(INTERIM_DATA_DIR / "bert_ready_data.csv")

# Load keywords from YAML file
with open(RAW_DATA_DIR / "classification-keyword.yaml", "r") as file:
    keywords = yaml.safe_load(file)

# Extract keyword categories
programming_languages = set(keywords["keywords"]["programming_languages"])
frameworks = set(keywords["keywords"]["frameworks"])
tools = set(keywords["keywords"]["tools"])
databases = set(keywords["keywords"]["databases"])


def combine_subwords(ner_results):
    """
    Combine subwords (that have ##) into full words
    """
    combined_results = []
    buffer_word = ""
    buffer_entity = None
    buffer_start = None

    for entity in ner_results:
        word = entity["word"]
        if word.startswith("##"):
            buffer_word += word[2:]  # Append to the current word
        else:
            if buffer_word:
                # Append the completed word before starting a new one
                combined_results.append(
                    {
                        "entity": buffer_entity,
                        "start": buffer_start,
                        "end": entity["start"],
                        "word": buffer_word,
                    }
                )
            buffer_word = word
            buffer_entity = entity["entity"]
            buffer_start = entity["start"]

    # Append the last word in the buffer
    if buffer_word:
        combined_results.append(
            {
                "entity": buffer_entity,
                "start": buffer_start,
                "end": ner_results[-1]["end"],
                "word": buffer_word,
            }
        )

    return combined_results


def refine_labels(ner_results):
    """
    Refine labels by adding specific categories (Programming Language, Framework, etc.)
    """
    refined_labels = []
    combined_results = combine_subwords(ner_results)

    for entity in combined_results:
        label = entity.get("entity")
        word = entity.get("word", "").strip().lower()

        # Map to specific categories based on keywords
        if label and (
            label.startswith("B-TECHNOLOGY") or label.startswith("I-TECHNOLOGY")
        ):
            if word in programming_languages:
                label = "PROGRAMMINGLANG"
            elif word in frameworks:
                label = "FRAMEWORK"
            elif word in tools:
                label = "TOOLS"
            elif word in databases:
                label = "DATABASE"
            else:
                label = "TECHNOLOGY"  # Default to TECHNOLOGY if no match
        else:
            # Skip words that are not TECHNOLOGY-related
            continue

        # Add refined label
        refined_labels.append(
            {
                "entity": label,
                "start": entity["start"],
                "end": entity["end"],
                "text": entity["word"],
            }
        )
    return refined_labels


# Create a list to store the results in Label Studio format
label_studio_data = []
all_entities = []  # To store all entities for counting

# Process each chunk
for index, row in df.iterrows():
    text = row["chunks"]
    ner_results = ner(text)
    refined_labels = refine_labels(ner_results)

    # Add refined labels to the Label Studio data
    label_studio_data.append(
        {
            "id": str(index),  # Ensure this is a string
            "data": {"text": text},  # Include text under "data"
            "annotations": [
                {
                    "id": index,  # Use an integer ID for the annotation
                    "result": [
                        {
                            "value": {
                                "start": label["start"],
                                "end": label["end"],
                                "text": label["text"],
                                "labels": [label["entity"]],
                            },
                            "id": f"result-{index}-{i}",  # Unique ID for result
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                        }
                        for i, label in enumerate(refined_labels)
                    ],
                }
            ],
        }
    )

    # Collect all entities for counting
    all_entities.extend([label["entity"] for label in refined_labels])

# Save the results as JSON
output_path = INTERIM_DATA_DIR / "labeled_ner_dataset.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_path}")

# Count the frequency of each unique entity
entity_counts = Counter(all_entities)

# Display unique entities and their counts
print("\nUnique entities and their counts:")
for entity, count in entity_counts.items():
    print(f"{entity}: {count}")
