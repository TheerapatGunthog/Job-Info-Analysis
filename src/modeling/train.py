import json
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import DatasetDict, Dataset
from pathlib import Path
import torch
import evaluate
from transformers import EarlyStoppingCallback

# Path ไปยังไฟล์ข้อมูล
TRAIN_FILE = "/home/whilebell/Code/Project/Job-Info-Analysis/data/processed/train_data_word_level.json"
VALIDATION_FILE = "/home/whilebell/Code/Project/Job-Info-Analysis/data/processed/validate_data_word_level.json"

# โมเดลที่ใช้ (เลือกโมเดลจาก Hugging Face)
MODEL_NAME = "distilbert-base-uncased"


# โหลดข้อมูล
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


train_data = load_data(TRAIN_FILE)
validation_data = load_data(VALIDATION_FILE)


# สร้าง Dataset
def convert_to_huggingface_dataset(data):
    return Dataset.from_list(data)


datasets = DatasetDict(
    {
        "train": convert_to_huggingface_dataset(train_data),
        "validation": convert_to_huggingface_dataset(validation_data),
    }
)

# โหลด Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ฟังก์ชันสำหรับ Tokenization และจัดเตรียม Input
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(labels[word_id])
        all_labels.append(aligned_labels)

        # Debugging
        # print(f"Example {i}:")
        # print(f"  Tokens: {examples['tokens'][i]}")
        # print(f"  Word IDs: {word_ids}")
        # print(f"  Original Labels: {labels}")
        # print(f"  Aligned Labels: {aligned_labels}")

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# Tokenize Dataset
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Debug ตัวอย่างข้อมูลหลัง Tokenization
print("Debug Tokenized Example:")
print(tokenized_datasets["train"][0])

# สร้าง label2id และ id2label
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

print("label2id:", label2id)
print("id2label:", id2label)

# โหลดโมเดล พร้อมกำหนด label2id และ id2label
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# ย้ายโมเดลไปยัง GPU (ถ้ามี)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics สำหรับการประเมินผล
metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)

    true_predictions = [
        [id2label[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print("Detailed Evaluation Metrics:", results)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# ตั้งค่าการ Train
training_args = TrainingArguments(
    output_dir="/home/whilebell/Code/Project/Job-Info-Analysis/models/testinglabelswithllm",
    evaluation_strategy="epoch",  # ประเมินผลทุก epoch
    save_strategy="epoch",  # บันทึกโมเดลทุก epoch
    learning_rate=1e-5,  # ค่าปกติสำหรับ fine-tune โมเดล transformer
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,  # เริ่มที่ 3-5 epochs
    weight_decay=0.01,  # Regularization
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,  # บันทึกโมเดลแค่ล่าสุด 2 checkpoints
    load_best_model_at_end=True,  # โหลดโมเดลที่ดีที่สุดหลังจบเทรน
    metric_for_best_model="f1",  # ใช้ f1-score เป็น metric หลัก
    greater_is_better=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # เพิ่ม EarlyStopping
)

# Train
print("Start Training...")
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

# บันทึกผลลัพธ์การ Evaluate
with open("evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=4)

# บันทึกโมเดลที่ฝึกแล้ว
model.save_pretrained(
    "/home/whilebell/Code/Project/Job-Info-Analysis/models/testinglabelswithllm/trained_ner_model"
)
tokenizer.save_pretrained(
    "/home/whilebell/Code/Project/Job-Info-Analysis/models/testinglabelswithllm/trained_ner_model"
)

# บันทึก Configuration
config = {"label2id": label2id, "id2label": id2label, "model_name": MODEL_NAME}
with open("model_config.json", "w") as f:
    json.dump(config, f, indent=4)

# ทดสอบ Inference
test_sentence = [
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
    ".",
]
inputs = tokenizer(test_sentence, return_tensors="pt", is_split_into_words=True).to(
    device
)
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)
predicted_labels = [
    id2label[id_] if id_ in id2label else "UNK" for id_ in predictions[0].cpu().numpy()
]
print("Test Sentence:", test_sentence)
print("tokens sentence:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
print("Predicted Labels:", predicted_labels)
