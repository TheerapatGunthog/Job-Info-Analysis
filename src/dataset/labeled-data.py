import pandas as pd
import yaml
from pathlib import Path
from transformers import pipeline

# Load the model
model_name = "GalalEwida/LLM-BERT-Model-Based-Skills-Extraction-from-jobdescription"
ner = pipeline("ner", model=model_name, tokenizer=model_name, device=0)

INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
RAW_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/raw")
df = pd.read_csv(INTERIM_DATA_DIR / "bert_ready_data.csv")

# Load keywords from YAML file
with open(RAW_DATA_DIR / "classification-keyword.yaml", "r") as file:
    keywords = yaml.safe_load(file)

# Use the 'chunks' column for processing
chunks = df["chunks"]

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
            buffer_word += word[2:]  # Continue the previous word
        else:
            if buffer_word:
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
    Refine labels by adding specific categories (Programming Language and Framework)
    """
    refined_labels = []
    combined_results = combine_subwords(ner_results)  # Combine words into full words

    for entity in combined_results:
        label = entity.get("entity")  # Get the label, if not present return None
        word = entity.get("word", "").strip().lower()

        # Check and adjust the label
        if label and (
            label.startswith("B-TECHNOLOGY") or label.startswith("I-TECHNOLOGY")
        ):
            if word in programming_languages:
                label = label.replace("B-TECHNOLOGY", "B-PROGRAMMINGLANG").replace(
                    "I-TECHNOLOGY", "I-PROGRAMMINGLANG"
                )
            elif word in frameworks:
                label = label.replace("B-TECHNOLOGY", "B-FRAMEWORK").replace(
                    "I-TECHNOLOGY", "I-FRAMEWORK"
                )
            elif word in tools:
                label = label.replace("B-TECHNOLOGY", "B-TOOLS").replace(
                    "I-TECHNOLOGY", "I-TOOLS"
                )
            elif word in databases:
                label = label.replace("B-TECHNOLOGY", "B-DATABASE").replace(
                    "I-TECHNOLOGY", "I-DATABASE"
                )

        else:
            label = "O"  # Set default value for words without entity or not matching conditions

        # Add the adjusted entry to the results
        refined_labels.append(
            {
                "entity": label,
                "start": entity.get(
                    "start", -1
                ),  # Set default value if start is not present
                "end": entity.get("end", -1),  # Set default value if end is not present
                "word": entity.get("word", ""),  # Keep the original word
            }
        )
    return refined_labels


# Create a list to store the results
result_data = []

# Process each chunk with the NER model
for chunk in chunks:
    ner_results = ner(chunk)

    # Refine labels
    refined_labels = refine_labels(ner_results)

    # Add chunk and refined labels to the list
    result_data.append({"chunks": chunk, "labels": refined_labels})

# Create a new DataFrame
new_dataset = pd.DataFrame(result_data)

# Save the results
output_path = INTERIM_DATA_DIR / "labeled_ner_dataset.csv"
new_dataset.to_csv(output_path, index=False)
