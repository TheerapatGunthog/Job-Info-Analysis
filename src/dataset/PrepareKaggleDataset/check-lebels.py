import pandas as pd
from collections import Counter

# โหลดข้อมูลจากไฟล์
df = pd.read_csv(
    "/home/whilebell/Code/Project/Job-Info-Analysis/data/external/ner_bert_data.csv"
)


# ฟังก์ชันสำหรับดึง labels ทั้งหมด
def extract_labels(labeled_tokens):
    tokens = eval(labeled_tokens)
    return [token[1] for token in tokens]


# ดึง labels ทั้งหมด
all_labels = []
for tokens in df["labeled_tokens"]:
    all_labels.extend(extract_labels(tokens))

# นับจำนวน unique labels
label_counts = Counter(all_labels)

# แสดงผลจำนวนการปรากฏของแต่ละ label
print("จำนวนครั้งที่แต่ละ Label ปรากฏ:")
print("-" * 40)
for label, count in label_counts.items():
    print(f"{label}: {count}")

# แสดง unique labels
print("\nUnique Labels ทั้งหมดที่พบ:")
print("-" * 40)
unique_labels = sorted(list(label_counts.keys()))
for i, label in enumerate(unique_labels, 1):
    print(f"{i}. {label}")

print(f"\nจำนวน Unique Labels ทั้งหมด: {len(label_counts)}")
