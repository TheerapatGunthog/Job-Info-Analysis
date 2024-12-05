from pathlib import Path
import yaml
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download("punkt")

# File Path
EXTERNAL_DATA_DIR = Path("../../../data/external")
RAW_DATA_DIR = Path("../../../data/raw")
PROCESSED_DATA_DIR = Path("../../../data/processed")

# data frame
df = pd.read_csv(EXTERNAL_DATA_DIR / "bert_ready_data.csv")


# ฟังก์ชันสำหรับโหลด labels keywords จากไฟล์ YAML
def load_labels_keywords(yaml_file):
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)
    return data.get("labels", {})


# โหลด labels keywords
yaml_file = RAW_DATA_DIR / "classification-keyword.yaml"
labels_keywords = load_labels_keywords(yaml_file)

# แยก keywords ตามหมวดหมู่
technology_keywords = labels_keywords.get("Technology", [])
role_keywords = labels_keywords.get("Role", [])
skill_keywords = labels_keywords.get("Skill", [])


# ฟังก์ชันสำหรับการติด BIO tags
def bio_tagger(tokens, keywords, tag_prefix):
    tags = ["O"] * len(tokens)
    for i, token in enumerate(tokens):
        for keyword in keywords:
            keyword_tokens = keyword.lower().split()
            if (
                token.lower() == keyword_tokens[0]
                and tokens[i : i + len(keyword_tokens)] == keyword_tokens
            ):
                tags[i] = f"B-{tag_prefix}"
                for j in range(1, len(keyword_tokens)):
                    tags[i + j] = f"I-{tag_prefix}"
    return tags


# ฟังก์ชันสำหรับการติดป้ายข้อมูล
def label_data(text):
    tokens = word_tokenize(text)
    tech_tags = bio_tagger(tokens, technology_keywords, "TECH")
    role_tags = bio_tagger(tokens, role_keywords, "ROLE")
    skill_tags = bio_tagger(tokens, skill_keywords, "SKILL")

    # Combine tags, giving priority to non-'O' tags
    final_tags = []
    for t, r, s in zip(tech_tags, role_tags, skill_tags):
        if t != "O":
            final_tags.append(t)
        elif r != "O":
            final_tags.append(r)
        elif s != "O":
            final_tags.append(s)
        else:
            final_tags.append("O")

    return list(zip(tokens, final_tags))


# ติดป้ายข้อมูล
df["labeled_tokens"] = df["description"].apply(label_data)

# บันทึกข้อมูลที่ติดป้ายแล้ว
df.to_csv(EXTERNAL_DATA_DIR / "ner_bert_data.csv", index=False)

# แสดงตัวอย่างผลลัพธ์
print(df["labeled_tokens"].head())
