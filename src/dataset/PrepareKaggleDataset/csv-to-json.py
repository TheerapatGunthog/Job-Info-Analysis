import json
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize

# File Path
EXTERNAL_DATA_DIR = Path("../../../data/external")
PROCESSED_DATA_DIR = Path("../../../data/processed")

# อ่านไฟล์ CSV ที่เราได้สร้างไว้ก่อนหน้านี้
df = pd.read_csv(EXTERNAL_DATA_DIR / "ner_bert_data.csv")


# ฟังก์ชันสำหรับแปลง BIO tags เป็นรูปแบบที่ต้องการ
def convert_bio_tags(bio_tag):
    if bio_tag == "O":
        return "O"
    elif bio_tag.startswith("B-"):
        return f"B-{bio_tag[2:]}"
    elif bio_tag.startswith("I-"):
        return f"I-{bio_tag[2:]}"
    else:
        return "O"  # กรณีที่ไม่รู้จัก tag


# สร้างลิสต์สำหรับเก็บข้อมูลในรูปแบบที่ต้องการ
dataset = []

# วนลูปผ่านแต่ละแถวในข้อมูล
for _, row in df.iterrows():
    text = row["description"]
    tokens = word_tokenize(text)

    labeled_tokens = eval(row["labeled_tokens"])  # แปลงสตริงเป็นลิสต์ของ tuples

    # สร้าง labels ให้มีความยาวเท่ากับ tokens
    labels = ["O"] * len(tokens)
    for i, (token, tag) in enumerate(labeled_tokens):
        if i < len(labels):
            labels[i] = convert_bio_tags(tag)

    entry = {"tokens": tokens, "labels": labels}

    dataset.append(entry)

# บันทึกเป็นไฟล์ JSON
with open(EXTERNAL_DATA_DIR / "ner_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print("บันทึกข้อมูลในรูปแบบ NER JSON เรียบร้อยแล้ว")

# แสดงตัวอย่างข้อมูล
print("\nตัวอย่างข้อมูล:")
print(json.dumps(dataset[:2], indent=2))
