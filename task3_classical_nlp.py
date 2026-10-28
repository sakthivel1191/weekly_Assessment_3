import re
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

Path("artifacts").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)
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
    for text in positive_texts:
        texts.append(f"{text} {['today', 'overall', 'for me'][index % 3]}")
        labels.append("positive")
    for text in negative_texts:
        texts.append(f"{text} {['today', 'overall', 'for me'][index % 3]}")
        labels.append("negative")
    for text in neutral_texts:
        texts.append(f"{text} {['today', 'overall', 'for me'][index % 3]}")
        labels.append("neutral")

df = pd.DataFrame({
    "text": texts,
    "label": labels
})
print("Dataset size:", len(df))
print(df["label"].value_counts())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
df["clean_text"] = df["text"].apply(clean_text)
print("\nExample cleaned text:")
print(df[["text", "clean_text"]].head())
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)
tfidf_model = Pipeline([
    ("tfidf",TfidfVectorizer(
        ngram_range=(1, 2),
            max_features=5000
        )
    ),
    ("classifier",LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

start_time = time.perf_counter()
tfidf_model.fit(X_train, y_train)
tfidf_training_time = time.perf_counter() - start_time
predictions = tfidf_model.predict(X_test)
tfidf_accuracy = accuracy_score(y_test, predictions)
tfidf_f1 = f1_score(
    y_test, predictions, average="macro"
)
print("\nTF-IDF Accuracy:", round(tfidf_accuracy, 4))
print("TF-IDF Macro F1:", round(tfidf_f1, 4))
print("\nClassification report:")
print(classification_report(y_test, predictions))
print("Confusion matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["negative", "neutral", "positive"]
))
vectorizer = tfidf_model.named_steps["tfidf"]
classifier = tfidf_model.named_steps["classifier"]
feature_names = vectorizer.get_feature_names_out()
print("\nImportant words for each class:")
for class_index, class_name in enumerate(classifier.classes_):
    coefficients = classifier.coef_[class_index]
    top_indices = np.argsort(coefficients)[-10:][::-1]
    top_words = feature_names[top_indices]
    print(f"\n{class_name}:")
    print(", ".join(top_words))

joblib.dump(
    tfidf_model,
    "artifacts/tfidf_pipeline.joblib"
)

try:
    import gensim.downloader as api
    print("\nLoading GloVe word vectors...")
    print("The first download can take some time.")
    glove = api.load("glove-wiki-gigaword-50")
    print("GloVe loaded successfully.")
    for word in ["king", "computer"]:
        if word in glove:
            print(f"\nFive words similar to '{word}':")
            print(glove.most_similar(word, topn=5))
    if all(word in glove for word in ["king", "man", "woman"]):
        result = glove.most_similar(
            positive=["king", "woman"],
            negative=["man"],
            topn=1
        )
        print("\nking - man + woman:", result)
    def sentence_vector(sentence):
        words = sentence.split()
        word_vectors = [
            glove[word]
            for word in words
            if word in glove
        ]
        if len(word_vectors) == 0:
            return np.zeros(50)
        return np.mean(word_vectors, axis=0)
    X_train_vectors = np.vstack([
        sentence_vector(text) for text in X_train
    ])
    X_test_vectors = np.vstack([
        sentence_vector(text) for text in X_test
    ])
    embedding_model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    start_time = time.perf_counter()
    embedding_model.fit(X_train_vectors, y_train)
    embedding_training_time = time.perf_counter() - start_time
    embedding_predictions = embedding_model.predict(
        X_test_vectors
    )
    embedding_accuracy = accuracy_score(
        y_test, embedding_predictions
    )
    embedding_f1 = f1_score(
        y_test,
        embedding_predictions,
        average="macro"
    )
    print("\nAveraged GloVe + Logistic Regression")
    print("Accuracy:", round(embedding_accuracy, 4))
    print("Macro F1:", round(embedding_f1, 4))
    print("\nTF-IDF versus averaged GloVe:")
    print(f"TF-IDF accuracy: {tfidf_accuracy:.4f}")
    print(f"GloVe accuracy:  {embedding_accuracy:.4f}")
    
except Exception as error:
    print("\nGloVe could not be loaded.")
    print("Reason:", error)
    print("TF-IDF results are still available.")
    print("Check internet access and the gensim installation.")
