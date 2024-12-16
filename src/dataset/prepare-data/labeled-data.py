import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline
from collections import Counter
import json
import re
from tqdm import tqdm

# Load the model
model_name = "GalalEwida/LLM-BERT-Model-Based-Skills-Extraction-from-jobdescription"
ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)

INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
RAW_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/raw")
df = pd.read_csv(INTERIM_DATA_DIR / "chunking_data.csv")

# Load keywords from YAML file
with open(RAW_DATA_DIR / "classification-keyword.yaml", "r") as file:
    classification_keywords = yaml.safe_load(file)

with open(RAW_DATA_DIR / "exclusion-keyword.yaml", "r") as file:
    exclusion_keywords = yaml.safe_load(file)


def flatten_and_lower(keyword_list):
    """
    Flatten a nested list of keywords and convert all words to lowercase.
    """
    flattened_list = []
    for item in keyword_list:
        if isinstance(item, list):  # ถ้าเป็น list ให้แยกออกมา
            flattened_list.extend([sub_item.lower() for sub_item in item])
        else:  # ถ้าเป็น string ธรรมดา
            flattened_list.append(item.lower())
    return set(flattened_list)


# Extract keyword categories
programming_languages = flatten_and_lower(
    classification_keywords["keywords"]["programming_languages"]
)
databases = flatten_and_lower(classification_keywords["keywords"]["databases"])
frameworkandlibary = flatten_and_lower(
    classification_keywords["keywords"]["frameworks_libraries"]
)
tools = flatten_and_lower(classification_keywords["keywords"]["tools"])
exclusion_word = flatten_and_lower(exclusion_keywords["keywords"])


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


def refine_labels(ner_results, text):
    """
    Refine labels by adding BIO scheme for multiword and single-word keywords.
    """
    refined_labels = []
    combined_results = combine_subwords(ner_results)

    # Helper: Build a multiword regex pattern
    def build_multiword_patterns(keyword_list, label_prefix):
        patterns = []
        for phrase in keyword_list:
            if " " in phrase:  # Detect multiword phrases only
                regex = r"\b" + r"\s+".join(map(re.escape, phrase.split())) + r"\b"
                patterns.append(
                    (re.compile(regex, re.IGNORECASE), label_prefix, phrase)
                )
        return patterns

    # Create multiword patterns for all categories
    multiword_patterns = []
    multiword_patterns += build_multiword_patterns(
        programming_languages, "PROGRAMMINGLANG"
    )
    multiword_patterns += build_multiword_patterns(databases, "DATABASE")
    multiword_patterns += build_multiword_patterns(
        frameworkandlibary, "FRAMEWORK_LIBRARY"
    )
    multiword_patterns += build_multiword_patterns(tools, "TOOLS")

    # Detect and label multiword entities
    matches = []
    for pattern, label_prefix, phrase in multiword_patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            words = phrase.split()
            current_pos = start

            for i, word in enumerate(words):
                word_start = text.find(word, current_pos, end)
                word_end = word_start + len(word)
                label = f"B-{label_prefix}" if i == 0 else f"I-{label_prefix}"
                matches.append(
                    {
                        "entity": label,
                        "start": word_start,
                        "end": word_end,
                        "text": text[word_start:word_end],
                    }
                )
                current_pos = word_end

    refined_labels.extend(matches)

    # Process single-word entities (already handled in YAML)
    for entity in combined_results:
        word = entity.get("word", "").strip().lower()
        if word in exclusion_word:
            continue

        label = None
        if word in programming_languages:
            label = "B-PROGRAMMINGLANG"
        elif word in databases:
            label = "B-DATABASE"
        elif word in frameworkandlibary:
            label = "B-FRAMEWORK_LIBRARY"
        elif word in tools:
            label = "B-TOOLS"

        if label:
            refined_labels.append(
                {
                    "entity": label,
                    "start": entity["start"],
                    "end": entity["end"],
                    "text": entity["word"],
                }
            )

    # Sort refined labels by start position (optional for clarity)
    refined_labels = sorted(refined_labels, key=lambda x: x["start"])

    return refined_labels


# Create a list to store the results in Label Studio format
label_studio_data = []
all_entities = []  # To store all entities for counting
idcount = 0
# Process each chunk
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing chunks"):
    text = row["chunks"]
    ner_results = ner(text)

    refined_labels = refine_labels(ner_results, text)

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
