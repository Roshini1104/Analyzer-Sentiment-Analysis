"""Train the four SentiScope sentiment classifiers and save them for app.py.

This reproduces STEP 2 of the original Colab notebook:
  Pipeline(TfidfVectorizer -> classifier) for each model, evaluated on a
  held-out 20% test split (stratified, random_state=42).

Each saved .joblib file is a full sklearn Pipeline, so the TF-IDF
vectorizer is bundled inside the model file — app.py can call
model.predict(["raw text"]) directly with no separate vectorizer file.

Usage:
    # Train on the built-in demo dataset (no CSV needed):
    python train_models.py

    # Train on your own CSV (must have a text column and a sentiment column):
    python train_models.py --csv my_dataset.csv --text-col text --label-col sentiment

Outputs (written next to this script):
    analyzer_models/random_forest.joblib
    analyzer_models/svm.joblib
    analyzer_models/logistic_regression.joblib
    analyzer_models/naive_bayes.joblib
    analyzer_metrics.csv
"""

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "analyzer_models"
METRICS_PATH = BASE_DIR / "analyzer_metrics.csv"

# ---------------------------------------------------------------------------
# Built-in demo dataset (used when no --csv is given).
# Diverse, realistic review/comment sentences per sentiment class.
# ---------------------------------------------------------------------------
DEMO_DATA = {
    "Positive": [
        "I absolutely love this product, it works perfectly and exceeded my expectations.",
        "Excellent quality and fast delivery, I will definitely buy again.",
        "The service was outstanding and the staff were very helpful and friendly.",
        "This is the best purchase I have made this year, highly recommended.",
        "Wonderful experience from start to finish, everything was perfect.",
        "The app is easy to use and the features are amazing, five stars.",
        "Great value for money and the quality is superb, very happy.",
        "I am extremely satisfied with this, it works flawlessly every day.",
        "Fantastic customer support, they solved my problem within minutes.",
        "The design is beautiful and the performance is lightning fast.",
        "Brilliant product, I recommend it to everyone who asks me.",
        "Superb quality, durable materials and a lovely finish, love it.",
        "My favourite purchase so far, the quality is truly impressive.",
        "Fast shipping, great packaging and the item is exactly as described.",
        "Easy to install and works beautifully, I am very pleased.",
        "The team went above and beyond to help me, thank you so much.",
        "Outstanding performance and reliability, worth every penny.",
        "I love the new update, it is faster and much more intuitive.",
        "Perfect fit, comfortable and stylish, I wear it every day.",
        "The food was delicious and the portions were generous, wonderful.",
        "Clean rooms, polite staff and a great location, highly recommended.",
        "The movie was entertaining and the acting was superb throughout.",
        "Brilliant battery life and a stunning display, best phone ever.",
        "The course content was clear, practical and extremely useful.",
        "Friendly teachers, engaging lessons and great results, thank you.",
        "The hotel exceeded our expectations, breakfast was fantastic.",
        "Smooth checkout process and instant confirmation, very impressed.",
        "The book was gripping from the first page to the last, loved it.",
        "Reliable, affordable and easy to use, exactly what I needed.",
        "The concert was magical and the sound quality was incredible.",
    ],
    "Negative": [
        "This product is terrible, it broke after just one week of use.",
        "Very disappointed, the quality is poor and it does not work as advertised.",
        "Awful experience, rude staff and a complete waste of money.",
        "The item arrived damaged and the replacement process was a nightmare.",
        "I regret buying this, it is slow, buggy and frustrating to use.",
        "Poor quality materials, it feels cheap and started falling apart.",
        "The service was delayed by days and nobody informed me about it.",
        "Terrible value, overpriced for such low quality and poor performance.",
        "The app keeps crashing and I have lost important data twice.",
        "Completely dissatisfied, the product failed during my first use.",
        "Horrible customer service, they ignored my emails for weeks.",
        "The food was cold, bland and the portion size was tiny.",
        "Dirty room, noisy neighbours and unhelpful reception, avoid this place.",
        "The movie was boring, predictable and poorly acted throughout.",
        "Battery drains in two hours, the worst phone I have ever owned.",
        "Confusing instructions and missing parts, impossible to assemble.",
        "The course was disorganised and the content was outdated and useless.",
        "Overbooked, understaffed and chaotic, a very stressful experience.",
        "The book was dull and repetitive, I could not finish it.",
        "Unreliable delivery, wrong item sent and no refund offered.",
        "The website is slow, checkout fails and support never replies.",
        "Cheap stitching, faded colours and a zip that broke immediately.",
        "The concert sound was distorted and the venue was overcrowded.",
        "Misleading advertising, the real product looks nothing like the photos.",
        "Frequent errors, data loss and a frustrating user interface.",
        "The staff were dismissive and the manager refused to help us.",
        "Cold soup, rude waiter and a forty minute wait for the bill.",
        "The update removed useful features and introduced new bugs.",
        "Fragile packaging, cracked screen on arrival, very disappointed.",
        "The lessons were rushed and the teacher was hard to understand.",
    ],
    "Neutral": [
        "The product arrived on time and matches the description in the photos.",
        "It is an average product, neither great nor terrible, it does the job.",
        "The service was standard, nothing special but nothing went wrong either.",
        "The item is okay for the price, though the packaging could be better.",
        "Delivery took three days, which is about what was promised.",
        "The app has the basic features I need, some are a bit slow.",
        "Quality is acceptable for everyday use, it is not premium grade.",
        "The room was clean and the location was convenient for our trip.",
        "The food was fine, portions were normal and prices were reasonable.",
        "The movie was entertaining enough, though the ending was predictable.",
        "Battery life is average, lasting about a full day of normal use.",
        "The course covered the basics clearly, though it lacked depth.",
        "Staff were polite and answered my questions without any issues.",
        "The book is informative but the writing style is quite dry.",
        "Checkout worked smoothly and I received my confirmation email quickly.",
        "The concert was enjoyable, the setlist included most of the hits.",
        "Materials feel sturdy enough, though the finish is fairly plain.",
        "The update added some features and changed a few existing ones.",
        "It fits as expected and the colour matches what was shown online.",
        "The hotel was quiet, breakfast was included and parking was available.",
        "The website layout is simple and I found what I needed easily.",
        "Performance is decent for regular tasks, though not the fastest.",
        "The teacher explained the concepts clearly and answered questions patiently.",
        "Packaging was adequate and nothing was damaged during delivery.",
        "The product works as described, with no surprises either way.",
        "Prices are mid-range and the quality reflects that fairly.",
        "The event was well organised and started on schedule.",
        "I have used it for a month now and it has been reliable so far.",
        "The manual was clear enough and setup took about ten minutes.",
        "Overall a reasonable purchase, I would consider buying again.",
    ],
}


def build_demo_dataframe() -> pd.DataFrame:
    import random

    rows = []
    for label, texts in DEMO_DATA.items():
        for text in texts:
            rows.append({"text": text, "sentiment": label})

    # Expand with template-generated samples so the demo models learn
    # class-specific vocabulary (keeps the demo metrics meaningful).
    rng = random.Random(42)
    subjects = [
        "the product", "the service", "the app", "the delivery",
        "the quality", "the support team", "the website", "the experience",
        "the purchase", "the item", "the hotel", "the movie",
        "the course", "the book", "the concert", "the food",
    ]
    pos_words = [
        "excellent", "fantastic", "wonderful", "amazing", "superb",
        "brilliant", "outstanding", "perfect", "delightful", "splendid",
        "lovely", "impressive", "exceptional", "marvellous", "superb",
    ]
    neg_words = [
        "terrible", "awful", "horrible", "dreadful", "pathetic",
        "appalling", "disgusting", "atrocious", "miserable", "dreadful",
        "poor", "useless", "frustrating", "disappointing", "shocking",
    ]
    neu_words = [
        "average", "standard", "ordinary", "typical", "moderate",
        "acceptable", "adequate", "fair", "normal", "regular",
        "decent", "reasonable", "passable", "mediocre", "ordinary",
    ]
    templates = {
        "Positive": [
            "This is {w} and I am very happy with it.",
            "Truly {w} quality, I recommend it to everyone.",
            "The {s} was {w} from start to finish.",
            "I love it, {w} value and {w} performance.",
            "What a {w} experience, five stars from me.",
        ],
        "Negative": [
            "This is {w} and I am very disappointed with it.",
            "Truly {w} quality, avoid buying this at all costs.",
            "The {s} was {w} from start to finish.",
            "I hate it, {w} value and {w} performance.",
            "What a {w} experience, I want my money back.",
        ],
        "Neutral": [
            "This is {w} and it does the job as expected.",
            "The {s} was {w}, nothing special either way.",
            "It is {w} for the price, neither good nor bad.",
            "The {s} is {w}, I have no strong opinion.",
            "Overall {w} experience, it is fine for everyday use.",
        ],
    }
    for label, words in [
        ("Positive", pos_words),
        ("Negative", neg_words),
        ("Neutral", neu_words),
    ]:
        for _ in range(120):
            template = rng.choice(templates[label])
            text = template.format(
                w=rng.choice(words), s=rng.choice(subjects)
            )
            rows.append({"text": text, "sentiment": label})

    return pd.DataFrame(rows)


def load_dataset(csv_path: str | None, text_col: str, label_col: str) -> pd.DataFrame:
    if csv_path:
        df = pd.read_csv(csv_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{text_col}' and '{label_col}'. "
                f"Found: {df.columns.tolist()}"
            )
        df = df[[text_col, label_col]].copy()
        df.columns = ["text", "sentiment"]
    else:
        df = build_demo_dataframe()
        print(f"Using built-in demo dataset: {len(df)} rows")

    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df = df[(df["text"] != "") & (df["sentiment"] != "")].copy()

    # Map common numeric / variant labels to Positive / Negative / Neutral.
    label_map = {
        "1": "positive", "pos": "positive", "positive": "positive",
        "0": "negative", "neg": "negative", "negative": "negative",
        "2": "neutral", "neu": "neutral", "neutral": "neutral",
    }
    df["sentiment"] = df["sentiment"].map(lambda v: label_map.get(v, v))
    df["sentiment"] = df["sentiment"].str.capitalize()

    # Drop classes with fewer than 2 samples (stratified split needs >= 2).
    counts = df["sentiment"].value_counts()
    rare = counts[counts < 2].index
    if len(rare) > 0:
        print(f"Dropping rare classes with <2 samples: {list(rare)}")
        df = df[~df["sentiment"].isin(rare)]

    if df["sentiment"].nunique() < 2:
        raise ValueError("Dataset must contain at least two sentiment classes.")
    print(f"Dataset shape: {df.shape}")
    print(df["sentiment"].value_counts().to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="Path to your training CSV.")
    parser.add_argument("--text-col", default="text", help="Text column name in the CSV.")
    parser.add_argument("--label-col", default="sentiment", help="Label column name in the CSV.")
    args = parser.parse_args()

    df = load_dataset(args.csv, args.text_col, args.label_col)
    X, y = df["text"], df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Same classifiers and TF-IDF settings as the original notebook.
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        "SVM": LinearSVC(random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    trained_models = {}
    results = []

    for name, classifier in classifiers.items():
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True, max_features=20000,
                        ngram_range=(1, 2), sublinear_tf=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        start = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - start

        predictions = model.predict(X_test)
        results.append(
            {
                "Model": name,
                "Accuracy (%)": round(accuracy_score(y_test, predictions) * 100, 2),
                "Precision (%)": round(
                    precision_score(y_test, predictions, average="macro", zero_division=0) * 100, 2
                ),
                "Recall (%)": round(
                    recall_score(y_test, predictions, average="macro", zero_division=0) * 100, 2
                ),
                "F1 Score (%)": round(
                    f1_score(y_test, predictions, average="macro", zero_division=0) * 100, 2
                ),
                "Training Time (sec)": round(train_seconds, 2),
            }
        )
        trained_models[name] = model
        print(f"\n{name}:\n{classification_report(y_test, predictions, zero_division=0)}")
        print(f"Confusion matrix:\n{confusion_matrix(y_test, predictions, labels=model.classes_)}")

    results_df = pd.DataFrame(results).sort_values("F1 Score (%)", ascending=False).reset_index(drop=True)
    print("\nMetrics:\n", results_df.to_string(index=False))
    print("\nBest model by macro F1:", results_df.iloc[0]["Model"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        out_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, out_path)
        print(f"Saved {out_path} (Pipeline includes TF-IDF vectorizer)")

    results_df.to_csv(METRICS_PATH, index=False)
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
