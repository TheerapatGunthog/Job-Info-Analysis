import pandas as pd
import re
from pathlib import Path
from nltk.corpus import stopwords

# โหลดไฟล์ CSV
EXTERNAL_DATA_DIR = Path("../../../data/external")
df = pd.read_csv(EXTERNAL_DATA_DIR / "filtered_job_descriptions.csv")


# ฟังก์ชันสำหรับ Clean ข้อความ
def clean_text(text):
    # 1. ลบ HTML Tags
    text = re.sub(r"<.*?>", "", text)
    # 2. ลบอักขระพิเศษ
    text = re.sub(r"[^a-zA-Z0-9.,'\" ]+", " ", text)
    # 3. แปลงเป็นตัวพิมพ์เล็ก
    text = text.lower()
    # 4. ลบช่องว่างเกิน
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ใช้ฟังก์ชัน Clean กับคอลัมน์ description
df["description"] = df["description"].apply(clean_text)

output_path = EXTERNAL_DATA_DIR / "cleaned_data.csv"
df.to_csv(output_path, index=False)
