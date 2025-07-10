# 🎬 IMDb Sentiment Analysis

A deep learning project that classifies IMDb movie reviews as **positive** or **negative** using neural networks. Built from scratch with TensorFlow as part of a machine learning portfolio targeting Data Scientist & ML Engineer roles.

---

## 💡 Motivation

I created this project to strengthen my understanding of **natural language processing (NLP)** and **deep learning model implementation**, especially in real-world text classification tasks like movie sentiment analysis. This also supports my long-term goal of working in the **media and entertainment industry**, where understanding audience feedback at scale is critical.

---

## 🧠 Model Overview

- **Task**: Binary sentiment classification
- **Dataset**: IMDb reviews (preprocessed)
- **Model**: 3-layer neural network with categorical cross-entropy loss
- **Framework**: TensorFlow + Keras (no pre-trained embeddings yet)
- **Baseline**: Bag-of-words + dense layers  
- **Next**: Fine-tune a BERT model for interpretability comparison

---

## 📊 Results

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | 87.2%     |
| Loss        | 0.29      |
| Optimizer   | Adam      |
| Epochs      | 50        |

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- NumPy / pandas
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📁 Project Structure
imdb-sentiment-analysis/
├── data/ # Sample data files
├── notebooks/ # Jupyter notebooks
├── src/ # Helper modules
├── README.md # This file
└── requirements.txt # Dependencies

---

## 🚀 How to Run

```bash
git clone https://github.com/MiReMiReMiLa/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb
```

## 🔭 Future Work
Add BERT-based classifier with HuggingFace Transformers

Include attention visualization and explainability

Deploy as a Streamlit web app

---

## 📚 Credentials
This project builds on knowledge from:

DeepLearning.AI's Deep Learning Specialization

DeepLearning.AI's Natural Language Processing Specialization

---

## 📬 Contact
Created by Jie Zhou | https://www.linkedin.com/in/margaret-zhou/
