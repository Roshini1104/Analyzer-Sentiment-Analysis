# SentiScope — Sentiment Analysis (Analyzer)

A machine-learning sentiment analysis platform built with **Python, Gradio, and scikit-learn**.
It classifies text as **Positive / Negative / Neutral**, compares four ML models, explains
predictions with LIME & SHAP, and can batch-process CSV datasets (optionally with Apache Spark).

> This project was prepared from the Google Colab notebook `analyzersentimentstrw (1).py`
> (STEP 9: Launch Website) and reorganised into a clean, deployable folder structure.

---

## Project structure

```
Analyzer/
│
├── app.py                      # Complete Gradio application (run this)
├── requirements.txt            # Python libraries
├── train_models.py             # (Re)train the 4 models + regenerate metrics
├── analyzer_models/            # Saved trained models (TF-IDF Pipelines)
│   ├── random_forest.joblib
│   ├── svm.joblib
│   ├── logistic_regression.joblib
│   └── naive_bayes.joblib
│
├── analyzer_metrics.csv        # Model evaluation results
│
├── assets/                     # Website design files
│   ├── logo.png                # App logo / hero illustration
│   └── style.css               # Custom stylesheet (loaded by app.py)
│
└── README.md                   # This file
```

**Model files include the TF-IDF vectorizer.** Each `.joblib` file is a full
`sklearn.Pipeline(TfidfVectorizer → classifier)`, so `app.py` can call
`model.predict(["raw text"])` directly — no separate vectorizer file is needed.

---

## Quick start (local)

```bash
cd Analyzer
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:7860** in your browser.

**Login (default demo credentials):**

| Username | Password |
|----------|----------|
| `admin`  | `SentiScope@123` |

You can also register a new account on the login screen (stored locally in SQLite).

**Security:** The login controls the interface display; it is not server-side access control for Gradio endpoints. Do not expose this app or sensitive data publicly without adding proper authentication and authorization. Change the default credentials before any deployment, and protect the SQLite database.

To change the admin credentials, set environment variables before launching:

```bash
SENTISCOPE_USERNAME=myuser SENTISCOPE_PASSWORD=mypass python app.py
```

---

## Run on Google Colab

1. Upload the whole `Analyzer/` folder to Google Drive (or GitHub).
2. In Colab, mount Drive and install dependencies:

```python
!pip install -r /content/drive/MyDrive/Analyzer/requirements.txt
```

3. Launch the app:

```python
import os
os.chdir("/content/drive/MyDrive/Analyzer")
!python app.py
```

This starts a local server inside Colab. It does not automatically create a public URL; to access it outside Colab, configure a suitable tunnel or deploy the project to a hosting platform.

---

## Deploy (Hugging Face Spaces / Render / Railway / Docker)

`app.py` already uses the deployment-ready launch block:

```python
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
```

So any platform that sets `PORT` will work out of the box:

- **Hugging Face Spaces:** push this folder, set SDK to Gradio, entry file `app.py`.
- **Render / Railway:** Build command `pip install -r requirements.txt`,
  Start command `python app.py`.
- **Docker:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]
```

---

## Retraining the models

The original trained model files and dataset were not supplied. The included models were trained on a small built-in demo dataset (450 rows, including 360 template-generated examples), so the app can run immediately. The metrics reflect a random holdout from that demo data and are **not a reliable measure of real-world performance**; similar templates can occur in both splits. Retrain and evaluate on a representative dataset before relying on predictions.

The saved pipelines were created with scikit-learn 1.6.1; the requirements pin that version for compatibility. Only load joblib files from trusted sources.

To retrain on **your own dataset**:

```bash
python train_models.py --csv your_dataset.csv --text-col text --label-col sentiment
```

Your CSV must contain a text column (e.g. `text`, `review`, `tweet`) and a label
column (e.g. `sentiment`, `label`, `polarity`). This regenerates:

- `analyzer_models/*.joblib` — the four TF-IDF pipelines
- `analyzer_metrics.csv` — Accuracy / Precision / Recall / F1 / Training Time

Training uses the same setup as the original notebook: stratified 80/20 split
(`random_state=42`), `TfidfVectorizer(lowercase=True, max_features=20000,
ngram_range=(1,2), sublinear_tf=True)`, and the four classifiers
Random Forest (150 trees), Linear SVM, Multinomial Naive Bayes, Logistic Regression.

---

## Features

| Tab | What it does |
|-----|--------------|
| 📊 Dashboard | Model performance cards, metrics table, and interactive charts |
| 🔍 Analyze | Single-text sentiment prediction with confidence |
| 📁 Upload | Batch CSV analysis, sentiment summaries, downloadable results |
| 🧠 ML Model | Model cards and accuracy comparison |
| 🤖 AI Model | LIME & SHAP word-level explanations (optional deps) |
| 🌐 BigData | Apache Spark distributed processing (optional dep) |
| ⚡ Performance | Full metrics graphs + evaluation on uploaded labeled data |
| ✨ About | Project overview, workflow, and technology stack |
| ⚙️ Settings | Theme, account, local data, and model options |

Optional features need extra packages (see `requirements.txt`):

```bash
pip install lime shap pyspark   # pyspark also needs Java installed
```

---

## Files explained

- **`app.py`** — the complete Gradio app: model loading, prediction helpers,
  charts, login/registration (SQLite), CSV analysis, LIME/SHAP explanations,
  Spark mode, and the UI. Paths are resolved relative to `app.py` (`BASE_DIR`),
  so it works from any working directory.
- **`assets/style.css`** — custom stylesheet. `app.py` loads this file if present
  (falling back to an embedded copy), so you can restyle the app without editing Python.
- **`assets/logo.png`** — logo / hero illustration (decoded from the app's embedded image).
- **`analyzer_metrics.csv`** — evaluation results shown on the Dashboard and Performance tabs.

---

## Troubleshooting

- **`RuntimeError: No trained models were found`** — make sure the four `.joblib`
  files are in `analyzer_models/`, or run `python train_models.py` first.
- **LIME/SHAP errors** — install the optional packages: `pip install lime shap`.
- **Spark errors** — install `pyspark` and Java, or leave the Spark checkbox off.
- **Port already in use** — run with a different port: `PORT=7861 python app.py`.
