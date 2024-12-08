from pathlib import Path
import yaml
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import BertTokenizer
import json

tqdm.pandas()

# Download necessary NLTK data
nltk.download("punkt")

# File Path
EXTERNAL_DATA_DIR = Path("../../../data/external")
RAW_DATA_DIR = Path("../../../data/raw")

df = pd.read_csv(EXTERNAL_DATA_DIR / "bert_ready_data.csv")


# Function to load labels keywords with subcategories
def load_labels_keywords(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # รวบรวมข้อมูลจากทุกหมวดหมู่และหัวข้อย่อย
    all_keywords = {}
    for main_category, subcategories in data["computer_engineering"].items():
        for item in subcategories:
            all_keywords[item] = main_category
    return all_keywords


# Load labels keywords
yaml_file = RAW_DATA_DIR / "classification-keyword.yaml"
labels_keywords = load_labels_keywords(yaml_file)

# Prepare tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Function to tag tokens with BIO schema
def bio_tagger(tokens, keyword, tag_prefix):
    tags = ["O"] * len(tokens)
    keyword_tokens = keyword.lower().split()

    for i, token in enumerate(tokens):
        if (
            token.lower() == keyword_tokens[0]
            and tokens[i : i + len(keyword_tokens)] == keyword_tokens
        ):
            tags[i] = f"B-{tag_prefix}"
            for j in range(1, len(keyword_tokens)):
                tags[i + j] = f"I-{tag_prefix}"
    return tags


# Function to label data
def label_data(text, labels_keywords):
    tokens = word_tokenize(text)
    final_tags = ["O"] * len(tokens)

    for keyword, category in labels_keywords.items():
        category_tags = bio_tagger(tokens, keyword, category.upper())
        for i, tag in enumerate(category_tags):
            if tag != "O":
                final_tags[i] = tag

    return list(zip(tokens, final_tags))


# Label data
df["labeled_tokens"] = df["chunks"].progress_apply(
    lambda x: label_data(x, labels_keywords)
)

df = df[["chunks", "labeled_tokens"]]

df.to_csv(EXTERNAL_DATA_DIR / "ner_bert_data.csv", index=False)

print(df["labeled_tokens"].head())

# Convert labeled data to Label Studio format
label_studio_data = []

for _, row in df.iterrows():
    text = row["chunks"]
    labeled_tokens = row["labeled_tokens"]

    annotations = []
    for token, label in labeled_tokens:
        if label != "O":
            annotations.append(
                {
                    "start": text.find(token),
                    "end": text.find(token) + len(token),
                    "text": token,
                    "labels": [label],
                }
            )

    label_studio_data.append(
        {
            "data": {"text": text},
            "annotations": [
                {
                    "result": [
                        {
                            "value": {
                                "start": ann["start"],
                                "end": ann["end"],
                                "text": ann["text"],
                                "labels": ann["labels"],
                            },
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "origin": "manual",
                        }
                        for ann in annotations
                    ]
                }
            ],
        }
    )

# Save to JSON file
with open(EXTERNAL_DATA_DIR / "label_studio_data.json", "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, ensure_ascii=False, indent=2)

print("Data prepared for Label Studio.")
