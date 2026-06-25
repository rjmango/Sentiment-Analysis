# Sentiment Analysis on Movie Reviews

A machine learning pipeline that classifies IMDB movie reviews as positive or negative. Three model architectures are compared — a linear baseline, an ensemble method, and a feedforward neural network with pre-trained word embeddings — to evaluate their trade-offs in text sentiment classification.

---

## Problem Statement

Given a movie review in plain text, predict whether the sentiment expressed is **positive** or **negative**. This is a binary classification problem evaluated on the [Large Movie Review Dataset (ACL 2011)](https://ai.stanford.edu/~amaas/data/sentiment/) by Maas et al.

---

## Dataset

| Property | Value |
|---|---|
| Source | Stanford AI Lab — Large Movie Review Dataset v1.0 |
| Total samples | 50,000 IMDB reviews |
| Class balance | 25,000 positive / 25,000 negative (perfectly balanced) |
| Split | 25,000 train / 25,000 test |

The balanced class distribution meant no resampling was needed, and accuracy is a reliable metric alongside F1-score.

---

## Pipeline Overview

```
Raw HTML Reviews
      │
      ▼
BeautifulSoup (HTML tag removal)
      │
      ▼
NLTK (stopword removal, text normalization)
      │
      ├──────────────────────────┐
      ▼                          ▼
TF-IDF Vectorization      GloVe Tokenization
(20,000 features)         (74,066 unique tokens → 200-length sequences)
      │                          │
      ├──────┬───────            ▼
      ▼      ▼           Embedding Matrix (20,000 × 100)
     LR     RF                   │
                                 ▼
                        Feedforward Neural Network
                        (Embedding → Dense 128 → Dropout 0.3 → Sigmoid)
```

---

## Models

### 1. Logistic Regression (TF-IDF)
A linear baseline using TF-IDF features with 20,000 vocabulary size. Fast, interpretable, and strong on high-dimensional sparse text representations.

### 2. Random Forest (TF-IDF)
An ensemble of decision trees on the same TF-IDF feature space, designed to capture non-linear patterns the linear model might miss.

### 3. Feedforward Neural Network (GloVe)
A lightweight neural network using pre-trained **100-dimensional GloVe embeddings** (Stanford NLP, 400,000 word vectors) loaded into a Keras Embedding layer. Architecture: Embedding → GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid). Trained for 10 epochs with Adam (lr=0.001).

---

## Results

| Model | Accuracy | F1-Score (weighted) | Precision | Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.8837** | **0.8837** | 0.88 | 0.88 |
| Random Forest | 0.8572 | 0.8572 | 0.86 | 0.86 |
| Neural Network (GloVe) | 0.7984 | 0.7983 | 0.80 | 0.80 |

### Key Findings

- **Logistic Regression was the top performer**, achieving 88.4% accuracy and a perfectly balanced F1-score across both classes — no bias toward positive or negative sentiment.
- **Random Forest performed competitively** at 85.7% but fell slightly behind LR, which is a known behaviour in very high-dimensional sparse TF-IDF spaces where linear separability is strong.
- **The GloVe neural network underperformed** classical models at 79.8%. The simple feedforward architecture likely couldn't leverage the semantic richness of GloVe embeddings — sequential models (RNNs, CNNs, or Transformers) would be better suited for capturing sentiment context from word order.
- **Classical TF-IDF + LR remains a strong baseline** for binary sentiment classification, especially when computational cost and interpretability matter.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas, NumPy |
| NLP preprocessing | NLTK, BeautifulSoup4 |
| Feature extraction | scikit-learn (TfidfVectorizer) |
| Classical ML | scikit-learn (LogisticRegression, RandomForestClassifier) |
| Deep learning | TensorFlow 2.19 / Keras |
| Embeddings | GloVe 6B 100d (Stanford NLP) |
| Visualization | matplotlib, seaborn, WordCloud |
| Environment | Google Colab |

---

## Project Structure

```
sentiment-analysis-imdb/
│
├── Dame_un_grroup.ipynb   # Main notebook (full pipeline)
├── README.md
└── requirements.txt       # (optional) pip dependencies
```

---

## How to Run

1. Clone the repo and open the notebook in Google Colab or Jupyter.
2. Run all cells in order — the notebook downloads the IMDB dataset and GloVe embeddings automatically.
3. No manual data download required.

```bash
# Optional: install dependencies locally
pip install nltk tensorflow scikit-learn beautifulsoup4 wordcloud matplotlib seaborn
```

---

## References

Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning word vectors for sentiment analysis.* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics, 142–150.

Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global vectors for word representation.* Proceedings of EMNLP 2014, 1532–1543.
