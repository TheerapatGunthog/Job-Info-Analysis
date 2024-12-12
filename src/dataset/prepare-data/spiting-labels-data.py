import json
from pathlib import Path

# กำหนด Input Path และ Output Directory
INTERIM_DATA_DIR = Path("/home/whilebell/Code/Project/Job-Info-Analysis/data/interim")
input_file = INTERIM_DATA_DIR / "labeled_by_code_data.json"
output_dir = INTERIM_DATA_DIR / "chunks-code-labels"
output_dir.mkdir(exist_ok=True)  # สร้างโฟลเดอร์หากยังไม่มี

# Load ข้อมูล JSON
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# กำหนดจำนวนตัวอย่างต่อไฟล์
chunk_size = 50000

# แบ่งข้อมูลออกเป็นไฟล์ย่อย
for i in range(0, len(data), chunk_size):
    chunk = data[i : i + chunk_size]  # ตัดข้อมูลเป็นชุดละ chunk_size
    output_file = output_dir / f"chunk_{i // chunk_size + 1}.json"  # ตั้งชื่อไฟล์
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(chunk, outfile, ensure_ascii=False, indent=4)

    print(f"Saved chunk {i // chunk_size + 1} to {output_file}")

print("\nการแยกไฟล์สำเร็จแล้ว! แต่ละไฟล์จะมีไม่เกิน 5,000 ตัวอย่าง")
