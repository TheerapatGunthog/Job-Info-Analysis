import json
from pathlib import Path
import re
import random

# Path ไปยังโฟลเดอร์ข้อมูล
PROCESSED_DATA_DIR = Path(
    "/home/whilebell/Code/Project/Job-Info-Analysis/data/processed"
)

# Path ไปยังไฟล์ข้อมูล
INPUT_FILE = Path(
    "/home/whilebell/Code/Project/Job-Info-Analysis/data/processed/project-30-at-2024-12-10-17-25-527b7663.json"
)
TRAIN_OUTPUT_FILE = PROCESSED_DATA_DIR / "train_data_word_level.json"
VALIDATION_OUTPUT_FILE = PROCESSED_DATA_DIR / "validate_data_word_level.json"

# สร้าง dictionary ของ label ไปยัง index
label2id = {
    "O": 0,
    "B-PROGRAMMINGLANG": 1,
    "I-PROGRAMMINGLANG": 2,
    "B-TECHNOLOGY": 3,
    "I-TECHNOLOGY": 4,
    "B-DATABASE": 5,
    "I-DATABASE": 6,
    "B-FRAMEWORKLIBARY": 7,
    "I-FRAMEWORKLIBARY": 8,
}
id2label = {v: k for k, v in label2id.items()}


def word_tokenize(text):
    """
    ใช้การแบ่งด้วยช่องว่างและลบเครื่องหมายพิเศษออก
    """
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def process_dataset(input_file, train_file, validation_file, split_ratio=0.8, seed=42):
    processed_data = []

    with open(input_file, "r") as f:
        data = json.load(f)

    for item in data:
        text = item["data"]["text"]  # ข้อความเต็ม
        entities = item["annotations"][0]["result"]  # NER entities

        # Tokenize ข้อความเต็มแบบ Word-Level
        tokens = word_tokenize(text)
        token_offsets = []
        current_pos = 0

        for token in tokens:
            start = text.find(token, current_pos)
            end = start + len(token)
            token_offsets.append((start, end))
            current_pos = end

        # สร้าง labels (ner_tags) เริ่มต้นเป็น "O" ทั้งหมด
        ner_tags = [label2id["O"]] * len(tokens)

        # Map entities ไปยัง tokens
        for entity in entities:
            start, end, label = (
                entity["value"]["start"],
                entity["value"]["end"],
                entity["value"]["labels"][0],
            )

            entity_start_assigned = False

            for idx, (token_start, token_end) in enumerate(token_offsets):
                if token_start >= start and token_end <= end:
                    # Assign B-label for the first token
                    if not entity_start_assigned:
                        ner_tags[idx] = label2id[f"B-{label}"]
                        entity_start_assigned = True
                    else:
                        # Assign I-label for subsequent tokens
                        ner_tags[idx] = label2id[f"I-{label}"]

        # เพิ่มข้อมูลที่แปลงแล้ว
        processed_data.append(
            {"id": str(item["id"]), "tokens": tokens, "ner_tags": ner_tags}
        )

    # แยกข้อมูลเป็น train และ validation
    random.seed(seed)
    random.shuffle(processed_data)
    split_index = int(len(processed_data) * split_ratio)
    train_data = processed_data[:split_index]
    validation_data = processed_data[split_index:]

    # บันทึกผลลัพธ์ลงไฟล์ JSON
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(validation_file, "w") as f:
        json.dump(validation_data, f, indent=4)

    print(f"\nTrain data saved to {train_file}")
    print(f"Validation data saved to {validation_file}")


# เรียกใช้ฟังก์ชัน
process_dataset(INPUT_FILE, TRAIN_OUTPUT_FILE, VALIDATION_OUTPUT_FILE)
