import spacy
from transformers import BertTokenizer
import pandas as pd
from pathlib import Path
from tqdm import tqdm

tqdm.pandas()

# โหลด SpaCy ภาษาอังกฤษสำหรับแบ่งประโยค
nlp = spacy.load("en_core_web_sm")

# โหลด Tokenizer ของ BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# ฟังก์ชันแบ่งประโยค
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


# ฟังก์ชันสำหรับ chunk ข้อความ
def chunk_sentences(sentences, max_tokens=512):
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(tokenizer.tokenize(sentence))

        # ถ้าเพิ่มประโยคนี้แล้วเกิน max_tokens ให้ปิด chunk ปัจจุบัน
        if current_length + sentence_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        # เพิ่มประโยคเข้า chunk ปัจจุบัน
        current_chunk.append(sentence)
        current_length += sentence_length

    # เพิ่ม chunk ที่เหลือ
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# โหลดไฟล์ CSV
EXTERNAL_DATA_DIR = Path("../../../data/external")
df = pd.read_csv(EXTERNAL_DATA_DIR / "cleaned_data.csv")

# แบ่งประโยคก่อน chunking พร้อมแสดงความคืบหน้า
df["sentences"] = df["description"].progress_apply(lambda x: split_sentences(x))

# Chunk ข้อความตามประโยคพร้อมความคืบหน้า
df["chunks"] = df["sentences"].progress_apply(
    lambda x: chunk_sentences(x, max_tokens=510)
)

# ระเบิด chunks ออกเป็นหลายแถว
df_exploded = df.explode("chunks").reset_index(drop=True)

# บันทึกไฟล์ผลลัพธ์
output_path = EXTERNAL_DATA_DIR / "bert_ready_data.csv"
df_exploded.to_csv(output_path, index=False)

print(f"Data saved to {output_path}")
