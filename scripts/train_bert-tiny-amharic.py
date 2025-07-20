from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report
import os

# === CONFIG ===
MODEL_NAME = "Davlan/bert-tiny-amharic"
DATA_PATH = "data/labeled/amharic_ner_conll_format.txt"
OUTPUT_DIR = "models/amharic-ner-tiny"
EPOCHS = 3

# === Load and Parse CoNLL Data ===
def read_conll(file_path):
    sentences, labels = [], []
    with open(file_path, encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                token, tag = line.split()
                tokens.append(token)
                tags.append(tag)
    return sentences, labels

sentences, tags = read_conll(DATA_PATH)
label_list = sorted(set(tag for seq in tags for tag in seq))
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": tags})

# === Tokenization and Label Alignment ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align(examples):
    tokenized = tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_str = label[word_idx]
                if word_idx != prev_word_idx:
                    label_ids.append(label2id[label_str])
                else:
                    if label_str.startswith("B-"):
                        label_str = label_str.replace("B-", "I-")
                    label_ids.append(label2id[label_str])
                prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(tokenize_and_align, batched=True)

# === Load Model ===
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# === Train ===
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    save_steps=1000,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# === Predict & Evaluate ===
predictions = trainer.predict(dataset)
preds = predictions.predictions.argmax(axis=-1)
true_labels = predictions.label_ids

true_tags = [[id2label[t] for t in seq if t != -100] for seq in true_labels]
pred_tags = [[id2label[p] for (p, t) in zip(pred, true_labels[i]) if t != -100] for i, pred in enumerate(preds)]

print("\n" + classification_report(true_tags, pred_tags))
