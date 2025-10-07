# 📌 Recommendation System using Collaborative Filtering & Matrix Factorization
COMPANY : CODTECH IT SOLUTIONS

NAME : GAURAV PANDEY

INTERN ID : CT04DY1426

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

## 📖 Project Overview

This project implements a **Recommendation System** from scratch using two popular techniques:

1. **Collaborative Filtering (User-based & Item-based)** – Recommends items by finding similar users or items using cosine similarity.
2. **Matrix Factorization (FunkSVD)** – Learns latent user and item features using gradient descent for better predictions on sparse data.

The system is designed to predict user ratings and generate top-N recommendations. It can work with **MovieLens datasets** or generate **synthetic data** if no dataset is provided.

## 🚀 Features

* User-based Collaborative Filtering
* Item-based Collaborative Filtering
* Matrix Factorization with Bias (FunkSVD)
* Evaluation Metrics:

  * RMSE (Root Mean Square Error)
  * Precision@K (ranking quality of recommendations)
* Works with MovieLens datasets or synthetic demo data

## 📂 Project Structure

```
CodTech_internship/
│── task4.py               # Main script (all-in-one recommender system)
│── README.md              # Documentation
│── ratings.csv (optional) # MovieLens dataset or custom dataset
```

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Recommendation-System.git
cd Recommendation-System
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
pandas
scikit-learn
```

---

## ▶️ Usage

### 1. Run with synthetic data (default)

```bash
python task4.py
```

### 2. Run with MovieLens dataset

Download [MovieLens dataset](https://grouplens.org/datasets/movielens/) (e.g., `ratings.csv`) and update in `task4.py`:

```python
DATA_PATH = "ratings.csv"
```

Then run:

```bash
python task4.py
```

---

## 📊 Sample Output

```
User-based CF RMSE: 1.02
Item-based CF RMSE: 0.97
Epoch 1/20 train RMSE: 1.05
...
Epoch 20/20 train RMSE: 0.87
FunkSVD Test RMSE: 0.92
```

---

## 📌 Evaluation Metrics

* **RMSE** – Measures accuracy of predicted ratings vs actual ratings.
* **Precision@K** – Measures ranking quality of recommendations (how many of the top-K recommended items are actually relevant).

---

## 🌟 Future Improvements

* Add hybrid (content + collaborative) recommendation.
* Implement implicit feedback (views, clicks) using `LightFM` or `implicit` libraries.
* Deploy as a REST API or Web App.

# OUTPUT
<img width="1859" height="1063" alt="Image" src="https://github.com/user-attachments/assets/afbf06e4-ff9a-4ca6-abf4-863ffbf13d71" />
