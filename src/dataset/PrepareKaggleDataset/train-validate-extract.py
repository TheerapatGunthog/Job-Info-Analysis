import json
import random
from pathlib import Path

EXTERNAL_DATA_DIR = Path("../../../data/external")
PROCESSED_DATA_DIR = Path("../../../data/processed")

# กำหนดเส้นทางไปยังไฟล์ต้นฉบับ
DATA_FILE = EXTERNAL_DATA_DIR / "ner_dataset.json"

# โหลดข้อมูลจากไฟล์ JSON
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# กำหนดสัดส่วน Train และ Validation
train_ratio = 0.8
random.seed(42)  # เพื่อให้ได้ผลลัพธ์เดิมเสมอ
random.shuffle(data)  # สุ่มลำดับของข้อมูล

# คำนวณจำนวนตัวอย่างใน Train และ Validation
train_size = int(len(data) * train_ratio)

# แยกข้อมูล
train_data = data[:train_size]
validation_data = data[train_size:]

# บันทึกข้อมูลแยกเป็นไฟล์ JSON
output_dir = PROCESSED_DATA_DIR  # โฟลเดอร์สำหรับบันทึกไฟล์
output_dir.mkdir(exist_ok=True)

with open(output_dir / "kaggle_train_data.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(output_dir / "kaggle_validation_data.json", "w", encoding="utf-8") as f:
    json.dump(validation_data, f, ensure_ascii=False, indent=2)

print("Train และ Validation dataset ถูกบันทึกเรียบร้อยแล้ว!")
