# 🎬 IMDB Sentiment Analysis (RNN, LSTM, GRU)

## 🧠 Overview
This project performs sentiment analysis on IMDB movie reviews using Deep Learning models like RNN, LSTM, and GRU. The model classifies reviews as **positive** or **negative**.

---

## 🎯 Objectives
- Clean and preprocess text data  
- Convert text into sequences  
- Apply padding  
- Train deep learning models (RNN, LSTM, GRU)  
- Evaluate performance  

---

## 📁 Dataset
Dataset used: IMDB Dataset  
Contains:
- review → text data  
- sentiment → positive / negative  

---

## ⚙️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- TensorFlow / Keras  

---

## 🔄 Steps

### 1. Load Data
```python
import pandas as pd
data = pd.read_csv('IMDB Dataset.csv')
