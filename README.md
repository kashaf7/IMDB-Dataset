# 🎬 IMDB Sentiment Analysis using RNN, LSTM and GRU

## 📌 Project Overview

This project performs sentiment analysis on IMDB movie reviews using deep learning techniques.

The goal is to classify movie reviews into two categories:

- **Positive**
- **Negative**

The project demonstrates the complete NLP and deep learning workflow, including text cleaning, tokenization, sequence padding, train-test splitting, and building recurrent neural network models using TensorFlow and Keras.

---

## 🎯 Objectives

- Load and explore the IMDB movie review dataset
- Clean and preprocess text data
- Convert text into numerical sequences
- Apply sequence padding
- Train deep learning models using RNN, LSTM, and GRU architectures
- Evaluate model performance on unseen test data

---

## 📂 Dataset

The project uses an IMDB movie review dataset containing movie reviews and their corresponding sentiment labels.

### Dataset Features

| Feature | Description |
|---|---|
| `review` | Text content of the movie review |
| `sentiment` | Sentiment label: positive or negative |

### Dataset Size

- **Total Records:** 35,631
- **Total Columns:** 2

### Target Labels

- `positive` → 1
- `negative` → 0

---

## ⚙️ Workflow

### 1️⃣ Data Loading

The dataset is loaded using Pandas.

```python
data = pd.read_csv(
    '/content/IMDB Dataset.csv',
    sep=',',
    on_bad_lines='skip',
    engine='python',
    quotechar='"'
)
