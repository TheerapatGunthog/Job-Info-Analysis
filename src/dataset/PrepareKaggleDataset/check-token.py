import pandas as pd
from pathlib import Path
from transformers import BertTokenizer

# โหลด Tokenizer ของ BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Path สำหรับไฟล์
EXTERNAL_DATA_DIR = Path("../../../data/external")
jobdescriptiondataset = (
    EXTERNAL_DATA_DIR / "job-description-dataset/job_descriptions.csv"
)
datasetglassdoor = EXTERNAL_DATA_DIR / "jobs-dataset-from-glassdoor/eda_data.csv"
linkedinjobposting = (
    EXTERNAL_DATA_DIR / "LinkedIn Job Postings (2023 - 2024)/postings.csv"
)
ustechnologyjobsondice = (
    EXTERNAL_DATA_DIR / "U.S. Technology Jobs on Dice.com/dice_com-job_us_sample.csv"
)

# รายชื่อไฟล์ CSV และคอลัมน์ที่สนใจ
file_columns = {
    jobdescriptiondataset: "Job Description",
    datasetglassdoor: "Job Description",
    linkedinjobposting: "description",
    ustechnologyjobsondice: "jobdescription",
}


# ฟังก์ชันสำหรับหาข้อความที่มีจำนวน token สูงสุด
def get_longest_token_entry(file_path, column_name, tokenizer):
    try:
        # อ่านข้อมูลจากไฟล์ CSV
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ("", 0)

    # ตรวจสอบว่าคอลัมน์มีอยู่ในไฟล์หรือไม่
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in {file_path}")
        return ("", 0)

    # แปลงคอลัมน์เป็น string และแปลงข้อความเป็น token
    df[column_name] = df[column_name].astype(str)

    # คำนวณจำนวน token สำหรับแต่ละข้อความในคอลัมน์
    token_counts = df[column_name].apply(lambda x: len(tokenizer.tokenize(x)))

    # หาข้อความที่มีจำนวน token สูงสุด
    max_token_count = token_counts.max()
    longest_entry = df[column_name][token_counts.idxmax()]

    return longest_entry, max_token_count


# วนลูปเพื่อแสดงข้อความที่มีจำนวน token สูงสุดในแต่ละไฟล์
for file_path, column_name in file_columns.items():
    longest_entry, max_token_count = get_longest_token_entry(
        file_path, column_name, tokenizer
    )
    print(f"File: {file_path.name}, Max Tokens: {max_token_count}")
    # print(f"Longest Entry (100 tokens preview): {longest_entry[:100]}...")
