import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline
from collections import Counter
import json
import re

# Load the model
model_name = "GalalEwida/LLM-BERT-Model-Based-Skills-Extraction-from-jobdescription"
ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)

INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
RAW_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/raw")
df = pd.read_csv(INTERIM_DATA_DIR / "chunking_data.csv")

# Load keywords from YAML file
with open(RAW_DATA_DIR / "classification-keyword.yaml", "r") as file:
    keywords = yaml.safe_load(file)

# Extract keyword categories
programming_languages = set(
    map(str.lower, keywords["keywords"]["programming_languages"])
)
databases = set(map(str.lower, keywords["keywords"]["databases"]))
frameworkandlibary = set(map(str.lower, keywords["keywords"]["FrameworkLibrary"]))


def normalize_word(word):
    """
    Normalize word to lowercase and remove special characters like ##.
    """
    word = word.lower()
    word = re.sub(r"[^a-z0-9]", "", word)  # Remove non-alphanumeric characters
    return word


def combine_subwords(ner_results):
    combined_results = []
    buffer_word = ""
    buffer_entity = None
    buffer_start = None

    for entity in ner_results:
        word = entity["word"]
        start = entity.get("start")  # Safely retrieve start
        if word.startswith("##"):
            buffer_word += word[2:]  # Append to the current word
        else:
            if buffer_word and buffer_start is not None:  # Ensure buffer_start is valid
                # Append the completed word before starting a new one
                combined_results.append(
                    {
                        "entity": buffer_entity,
                        "start": buffer_start,
                        "end": buffer_start + len(buffer_word),
                        "word": buffer_word,
                    }
                )
            buffer_word = word
            buffer_entity = entity.get("entity")
            buffer_start = start  # Set buffer_start to a valid start position or None

    # Append the last word in the buffer
    if buffer_word and buffer_start is not None:  # Ensure buffer_start is valid
        combined_results.append(
            {
                "entity": buffer_entity,
                "start": buffer_start,
                "end": buffer_start + len(buffer_word),
                "word": buffer_word,
            }
        )
    return combined_results


def refine_labels(ner_results):
    """
    Refine labels by adding specific categories (Programming Language, Framework&&Libary, etc.)
    """
    refined_labels = []
    combined_results = combine_subwords(ner_results)

    print("\nCombined Results:")
    for entity in combined_results:
        print(entity)

    for entity in combined_results:
        label = entity.get("entity")
        word = entity.get("word", "").strip()
        normalized_word = normalize_word(word)

        # Debugging step to see matching process
        print(f"Checking word: '{normalized_word}'")

        # Map to specific categories based on keywords
        if (
            label
            and (label.startswith("B-TECHNOLOGY") or label.startswith("I-TECHNOLOGY"))
            or (label.startswith("B-TECHNICAL") or label.startswith("I-TECHNICAL"))
        ):
            if normalized_word in programming_languages:
                print(f"Matched '{normalized_word}' as PROGRAMMINGLANG")
                label = "PROGRAMMINGLANG"
            elif normalized_word in databases:
                print(f"Matched '{normalized_word}' as DATABASE")
                label = "DATABASE"
            elif normalized_word in frameworkandlibary:
                print(f"Matched '{normalized_word}' as FRAMEWORK")
                label = "FRAMEWORK&LIBARY"
            else:
                print(
                    f"No match found for '{normalized_word}', defaulting to TECHNOLOGY"
                )
                if label == "B-TECHNICAL" or label == "I-TECHNICAL":
                    continue
                else:
                    label = "TECHNOLOGY"
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
idcount = 0
# Process each chunk
for index, row in df.iterrows():
    text = row["chunks"]
    print(f"\nProcessing chunk {index}: {text}")
    ner_results = ner(text)

    # Debugging: Print raw NER results
    print("\nRaw NER results:")
    for r in ner_results:
        print(r)

    refined_labels = refine_labels(ner_results)

    # Add refined labels to the Label Studio data
    if refined_labels:  # ตรวจสอบว่า refined_labels ไม่ใช่ list ว่าง
        label_studio_data.append(
            {
                "id": str(idcount),  # Ensure this is a string
                "data": {"text": text},  # Include text under "data"
                "annotations": [
                    {
                        "id": idcount,  # Use an integer ID for the annotation
                        "result": [
                            {
                                "value": {
                                    "start": label["start"],
                                    "end": label["end"],
                                    "text": label["text"],
                                    "labels": [label["entity"]],
                                },
                                "id": f"result-{idcount}-{i}",  # Unique ID for result
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
        idcount += 1

    # Collect all entities for counting
    all_entities.extend([label["entity"] for label in refined_labels])

    # Collect all entities for counting
    all_entities.extend([label["entity"] for label in refined_labels])

# Save the results as JSON
output_path = INTERIM_DATA_DIR / "labeled_by_code_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, ensure_ascii=False, indent=4)

print(f"Data saved to {output_path}")

# Count the frequency of each unique entity
entity_counts = Counter(all_entities)

# Display unique entities and their counts
print("\nUnique entities and their counts:")
for entity, count in entity_counts.items():
    print(f"{entity}: {count}")
