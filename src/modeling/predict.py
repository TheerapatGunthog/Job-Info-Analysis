import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# Path ไปยังไฟล์โมเดลและ Configuration
MODEL_DIR = "/home/whilebell/Code/Project/Job-Info-Analysis/models/testinglabelswithllm/trained_ner_model"
CONFIG_FILE = (
    "/home/whilebell/Code/Project/Job-Info-Analysis/src/modeling/model_config.json"
)

# โหลด Configuration
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

label2id = config["label2id"]
id2label = {int(k): v for k, v in config["id2label"].items()}

# โหลด Tokenizer และ Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

# ย้ายโมเดลไปยัง GPU (ถ้ามี)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


# ฟังก์ชันสำหรับการ Predict
def predict(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        is_split_into_words=True,
        padding=True,
        truncation=True,
    ).to(device)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

    # Mapping ID ของ Labels เป็นชื่อ Labels
    predicted_labels = [
        id2label.get(id_, "UNK") for id_ in predictions[0].cpu().numpy()
    ]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # จัดรูปผลลัพธ์การพยากรณ์
    results = []
    for token, label in zip(tokens, predicted_labels):
        if token not in ["[PAD]", "[CLS]", "[SEP]"]:
            results.append((token, label))

    return results


# ตัวอย่างการใช้งาน
new_sentence = [
    "kotlin",
    "is",
    "a",
    "popular",
    "programming",
    "language",
    "mongodb",
    "and",
    "tensorflow",
    "are",
    "popular",
    "technologies",
    "spring",
    "and",
    "django",
    "are",
    "popular",
    "frameworks",
    "javascript",
    "is",
    "a",
    "popular",
    "programming",
    "language",
    "react",
    "and",
    "angular",
    "are",
    "popular",
    "frameworks",
    "tableau",
    "is",
    "a",
    "popular",
    "data",
    "visualization",
    "tool",
    "docker",
    "is",
    "a",
    "popular",
    "containerization",
    ".",
]

predictions = predict(new_sentence)

print("Predicted Results:")
for token, label in predictions:
    print(f"{token}: {label}")

inputs = tokenizer(new_sentence, return_tensors="pt", is_split_into_words=True).to(
    device
)
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predicted_labels = [
    id2label[id_] if id_ in id2label else "UNK" for id_ in predictions[0].cpu().numpy()
]

# รวมคำและ Label สำหรับ Subword Tokens
final_tokens = []
final_labels = []

current_token = ""
current_label = None

for token, label in zip(tokens, predicted_labels):
    if token.startswith("##"):
        current_token += token[2:]
    else:
        if current_token:
            final_tokens.append(current_token)
            final_labels.append(current_label)

        current_token = token
        current_label = label

# เก็บคำสุดท้าย
if current_token:
    final_tokens.append(current_token)
    final_labels.append(current_label)

# กรอง Token พิเศษออก
filtered_tokens_labels = [
    (token, label)
    for token, label in zip(final_tokens, final_labels)
    if token not in ["[CLS]", "[SEP]", "[PAD]"]
]

print("\nFiltered Results:")
for token, label in filtered_tokens_labels:
    print(f"{token}: {label}")
