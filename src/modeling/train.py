import json
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers import DataCollatorForTokenClassification
from datasets import Dataset
import evaluate
import numpy as np

# Define label mapping
label_mapping = {
    "O": 0,
    "B-TECHNOLOGY": 1,
    "I-TECHNOLOGY": 2,
    "B-SKILLS": 3,
    "I-SKILLS": 4,
    "B-CERTIFICATIONS": 5,
    "I-CERTIFICATIONS": 6,
    "B-SOFT_SKILLS": 7,
    "I-SOFT_SKILLS": 8,
    "B-EXPERIENCE_LEVEL": 9,
    "I-EXPERIENCE_LEVEL": 10,
    "B-DOMAIN_KNOWLEDGE": 11,
    "I-DOMAIN_KNOWLEDGE": 12,
    "B-EDUCATIONAL_BACKGROUND": 13,
    "I-EDUCATIONAL_BACKGROUND": 14,
}

# Load datasets
with open("../../data/processed/ner_train_dataset.json") as f:
    train_data = json.load(f)

with open("../../data/processed/ner_validation_dataset.json") as f:
    val_data = json.load(f)


# Ensure labels are integers using the label mapping
def convert_labels_to_int(data, label_mapping):
    for item in data:
        item["tags"] = [label_mapping[label] for label in item["tags"]]
    return data


train_data = convert_labels_to_int(train_data, label_mapping)
val_data = convert_labels_to_int(val_data, label_mapping)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_mapping)
)


# Tokenize datasets
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# Define data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load metric
metric = evaluate.load("seqeval")


# Example of using the metric
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label for label in label if label != -100] for label in labels]
    true_predictions = [
        [pred for pred, label in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
# Save the model
trainer.save_model("./ner-bert-model")

# Save the tokenizer
tokenizer.save_pretrained("./ner-bert-model")
