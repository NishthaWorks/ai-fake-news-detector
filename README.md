# 📰 Fake News Detector using Machine Learning (XGBoost + Streamlit)

Welcome to my first real-world AI project!  
This app detects whether a news article is **Real** or **Fake** using **Natural Language Processing (NLP)** and **XGBoost** — all wrapped in a simple **Streamlit web app**.

---

## 🚀 Project Overview

Fake news is one of the major challenges in the digital world.  
This project aims to classify news articles as **real** or **fake** based on their text content using NLP preprocessing and a machine learning classifier.

---

## 🧠 Tech Stack

- 🐍 **Python 3**
- 📊 **Pandas**, **NumPy**
- 🔠 **NLTK** (text cleaning & tokenization)
- 💡 **TF-IDF Vectorization**
- ⚙️ **XGBoost Classifier**
- 🌐 **Streamlit** (for web interface)
- 💾 **Joblib** (for saving models)

---

## 📊 Model Performance

| Metric     | Score |
|-------------|-------|
| Accuracy    | 99.7% |
| Precision   | 1.00  |
| Recall      | 1.00  |
| F1-Score    | 1.00  |

The model performs exceptionally well on unseen data and correctly identifies fake and real news almost perfectly.

---

## 🧹 Data Preprocessing Steps

1. Convert text to lowercase  
2. Tokenize words  
3. Remove stopwords and punctuation  
4. Lemmatize tokens  
5. Apply **TF-IDF Vectorization**

---

## 🧪 How to Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/your-username/fake-news-detector-nishtha.git
cd fake-news-detector-nishtha
