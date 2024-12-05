from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers import DataCollatorForTokenClassification
from pathlib import Path

PROCESSED_DATA_DIR = Path("../../data/processed")
MODEL_DIR = Path("../../models")
train_data_file = str(PROCESSED_DATA_DIR / "kaggle_train_data.json")
validation_data_file = str(PROCESSED_DATA_DIR / "kaggle_validation_data.json")
output_dir = str(MODEL_DIR / "results")

# Step 1: โหลด Dataset
# ใช้ JSON file ของคุณแทน path ด้านล่าง
data_files = {
    "train": train_data_file,  # ใส่ path ไปยังไฟล์ JSON ของ training set
    "validation": validation_data_file,  # ใส่ path ไปยัง validation set
}
dataset = load_dataset("json", data_files=data_files)

# Step 2: โหลด Tokenizer และ Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Step 3: Tokenization และ Alignment ของ Labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=True
    )
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # -100 คือ token ที่ไม่ถูกนำไป train
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # label ของคำแรก
            else:
                label_ids.append(-100)  # subword tokens ไม่สนใจ label
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Map function กับ dataset
label_list = [
    "O",
    "B-ROLE",
    "I-ROLE",
    "B-SKILL",
    "I-SKILL",
    "B-TECH",
    "I-TECH",
]  # กำหนด labels ทั้งหมด
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for i, label in enumerate(label_list)}


def convert_labels_to_ids(examples):
    examples["labels"] = [
        [label_to_id[label] for label in labels] for labels in examples["labels"]
    ]
    return examples


dataset = dataset.map(convert_labels_to_ids, batched=True)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Step 4: โหลด Model
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list)
)

# Step 5: Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Step 6: Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Step 7: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train
trainer.train()

metrics = trainer.evaluate()
print(metrics)
