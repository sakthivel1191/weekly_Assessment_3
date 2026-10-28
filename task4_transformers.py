import re
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset

positive_texts = [
    "I love this product",
    "Excellent quality and great service",
    "Very happy with my purchase",
    "The experience was wonderful",
    "Delivery was fast and easy",
    "The support team was helpful",
    "This product works perfectly",
    "Amazing value for money",
    "I would recommend this service",
    "Everything was smooth and reliable",
    "The app is easy to use",
    "Very satisfied with the result"
]
negative_texts = [
    "I hate this product",
    "Terrible quality and poor service",
    "Very unhappy with my purchase",
    "The experience was disappointing",
    "Delivery was late and difficult",
    "The support team was unhelpful",
    "This product does not work",
    "Very poor value for money",
    "I would not recommend this service",
    "Everything was confusing and unreliable",
    "The app is difficult to use",
    "Very dissatisfied with the result"
]
neutral_texts = [
    "The product arrived today",
    "The service is available",
    "I made a purchase yesterday",
    "The delivery took two days",
    "The support team answered my question",
    "The product has several features",
    "The price is listed on the website",
    "I received an order confirmation",
    "The application has a new update",
    "The package contains three items",
    "The account was created successfully",
    "The report contains the requested details"
]
texts = []
labels = []
for index in range(10):
    suffix = ["today", "overall", "for me"][index % 3]
    for text in positive_texts:
        texts.append(f"{text} {suffix}")
        labels.append("positive")
    for text in negative_texts:
        texts.append(f"{text} {suffix}")
        labels.append("negative")

    for text in neutral_texts:
        texts.append(f"{text} {suffix}")
        labels.append("neutral")
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

df = pd.DataFrame({
    "text": [clean_text(text) for text in texts],
    "label": labels
})
label_to_id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
df["label_id"] = df["label"].map(label_to_id)
model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
examples = [
    "I love this product.",
    "The delivery was late and disappointing."
]
print("\nTokenizer output:")
for sentence in examples:
    result = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=32
    )
    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(
        result["input_ids"]
    ))
    print("Input IDs:", result["input_ids"])
    print("Attention mask:", result["attention_mask"])

print("\nPretrained sentiment-analysis pipeline:")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)
sentiment_sentences = [
    "The service was excellent.",
    "The product was terrible.",
    "The package arrived today."
]
for result in sentiment_pipeline(sentiment_sentences):
    print(result)

print("\nPretrained NER pipeline:")
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)
ner_sentences = [
    "John works at Microsoft in London.",
    "Apple opened an office in California.",
    "Mary visited Paris last Monday."
]
for sentence in ner_sentences:
    print("\n", sentence)
    print(ner_pipeline(sentence))

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label_id"]
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.20,
    random_state=42,
    stratify=train_df["label_id"]
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=64
        )
        self.labels = list(labels)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        item = {
            key: torch.tensor(value[index])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long
        )
        return item
train_dataset = TextDataset(
    train_df["text"], train_df["label_id"], tokenizer
)
val_dataset = TextDataset(
    val_df["text"], val_df["label_id"], tokenizer
)
test_dataset = TextDataset(
    test_df["text"], test_df["label_id"], tokenizer
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={
        0: "negative",
        1: "neutral",
        2: "positive"
    },
    label2id=label_to_id
)

def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels, predictions, average="macro"
        )
    }
Path("artifacts/transformer").mkdir(
    parents=True, exist_ok=True
)
training_args = TrainingArguments(
    output_dir="artifacts/training_checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available()
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
start_time = time.perf_counter()
trainer.train()
transformer_training_time = time.perf_counter() - start_time
test_results = trainer.evaluate(test_dataset)
print("\nTransformer test results:")
print(test_results)
trainer.save_model("artifacts/transformer")
tokenizer.save_pretrained("artifacts/transformer")

with open(
    "artifacts/transformer/label_mapping.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(label_to_id, file, indent=2)

tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])
start_time = time.perf_counter()
tfidf_model.fit(train_df["text"], train_df["label"])
tfidf_training_time = time.perf_counter() - start_time
tfidf_predictions = tfidf_model.predict(test_df["text"])
tfidf_accuracy = accuracy_score(
    test_df["label"], tfidf_predictions
)
tfidf_f1 = f1_score(
    test_df["label"],
    tfidf_predictions,
    average="macro"
)
test_texts = test_df["text"].tolist()
trainer.model.eval()
sample_inputs = tokenizer(
    test_texts[:2],
    truncation=True,
    padding=True,
    return_tensors="pt"
)
device = next(trainer.model.parameters()).device
sample_inputs = {
    key: value.to(device)
    for key, value in sample_inputs.items()
}

with torch.no_grad():
    trainer.model(**sample_inputs)
start_time = time.perf_counter()
tfidf_model.predict(test_texts)
tfidf_inference_time = (
    time.perf_counter() - start_time
) / len(test_texts)
start_time = time.perf_counter()
trainer.predict(test_dataset)
transformer_inference_time = (
    time.perf_counter() - start_time
) / len(test_texts)
transformer_accuracy = test_results["eval_accuracy"]
transformer_f1 = test_results["eval_macro_f1"]
comparison = pd.DataFrame([
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": tfidf_accuracy,
        "Macro F1": tfidf_f1,
        "Training time (seconds)": tfidf_training_time,
        "Inference time (seconds/example)": tfidf_inference_time
    },
    {
        "Model": "Fine-tuned DistilBERT",
        "Accuracy": transformer_accuracy,
        "Macro F1": transformer_f1,
        "Training time (seconds)": transformer_training_time,
        "Inference time (seconds/example)": transformer_inference_time
    }
])
print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))
comparison.to_csv(
    "reports/comparison.csv",
    index=False
)

