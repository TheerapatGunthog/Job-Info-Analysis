import pandas as pd
import json
import ast
from sklearn.model_selection import train_test_split
from pathlib import Path

# กำหนด Path
EXTERNAL_DATA_DIR = Path("../../../data/external")
PROCESSED_DATA_DIR = Path("../../../data/processed")

# อ่านข้อมูล
df = pd.read_csv(EXTERNAL_DATA_DIR / "ner_bert_data.csv")

# แปลง string เป็น list
df["labeled_tokens"] = df["labeled_tokens"].apply(ast.literal_eval)


def convert_to_ner_format(row):
    tokens, tags = zip(*row["labeled_tokens"])
    return {"tokens": list(tokens), "tags": list(tags)}


# แปลงข้อมูลเป็นรูปแบบสำหรับ NER
ner_data = df.apply(convert_to_ner_format, axis=1).tolist()

# แบ่งข้อมูล train/validate (80:20)
train_data, val_data = train_test_split(ner_data, test_size=0.2, random_state=42)

# สร้างโฟลเดอร์ถ้ายังไม่มี
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# บันทึกข้อมูล train เป็น JSON
with open(PROCESSED_DATA_DIR / "ner_train_dataset.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

# บันทึกข้อมูล validation เป็น JSON
with open(
    PROCESSED_DATA_DIR / "ner_validation_dataset.json", "w", encoding="utf-8"
) as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

# แสดงตัวอย่างข้อมูล train:
print("ตัวอย่างข้อมูล train:")
print(json.dumps(train_data[0], indent=2))
print(f"\nจำนวนข้อมูล train: {len(train_data)}")
print(f"จำนวนข้อมูล validation: {len(val_data)}")
