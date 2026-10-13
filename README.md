# Autism Screening Prediction App

A Flask + MySQL web application that predicts whether a child's answers to the AQ-10 autism screening questionnaire indicate autism spectrum disorder (ASD) traits, together with a comparison of machine-learning models on the same data.

B.Tech final-year group project (team of 4) at Sri Venkateswara University, 2025. This repository is a cleaned-up, re-evaluated version of the project code.

> **Disclaimer:** this is an academic project. It is not a diagnostic tool and must not be used for medical decisions.

## Features

- User registration and login (MySQL, salted password hashes, session-based auth)
- Upload a dataset (CSV or Excel), choose a train/test split, and compare models
- For each model, shows held-out test accuracy **and** 5-fold cross-validated accuracy, computed live
- Prediction form: answer the 10 AQ-10 questions plus age, sex, jaundice at birth, and family history of autism, and get a screening result from a linear SVM

## Dataset

[Autistic Spectrum Disorder Screening Data for Children](https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children) from the UCI Machine Learning Repository (Fadi Thabtah, licensed CC BY 4.0), ages 4–11. Stored at `data/autism_child_data.csv`.

Features used: the 10 AQ-10 answers (0/1), age, sex, jaundice at birth, and family member with ASD.

## Results

Run `python evaluate.py` to reproduce:

| Model | 70/30 split (raw data) | 5-fold CV (deduplicated) |
|---|---|---|
| SVM (linear) | 100.0% | 100.0% ± 0.0% |
| Logistic Regression | 100.0% | 100.0% ± 0.0% |
| Random Forest | 100.0% | 94.3% ± 3.0% |
| Decision Tree | 100.0% | 87.8% ± 3.4% |
| XGBoost | 100.0% | 95.1% ± 2.0% |

The original notebook (`notebooks/01_model_comparison.ipynb`) also trained an ANN (98.0% test accuracy), a 1D-CNN (93.3%) and an LSTM (93.3%) with Keras.

### Why are the scores so high?

Two properties of this dataset explain it, and they are worth knowing before trusting any result on it:

1. **Duplicates.** 250 of the 498 rows are exact duplicates. With a random split, copies of the same record land in both train and test sets, which inflates test accuracy. That is why every model reaches 100% on the raw split, while the tree-based models drop to 88–95% once duplicates are removed and cross-validation is used.
2. **The label is a scoring rule.** The `Class/ASD` label equals "YES" exactly when the AQ-10 score (the sum of the 10 answers) is 7 or more. A one-line threshold rule therefore scores 100%. Linear models can represent this rule exactly; tree models only approximate it.

So the models reproduce the questionnaire's scoring threshold; they don't discover new diagnostic signal. A meaningful next step would be predicting a clinical diagnosis rather than the screening label.

## Project structure

```
app.py            Flask routes (auth, upload, split, model comparison, prediction)
ml.py             Data cleaning, model definitions, evaluation helpers
evaluate.py       Reproduces the results table above
db.sql            MySQL schema
data/             UCI children's ASD screening dataset
notebooks/        Original exploratory notebook (ML + ANN/CNN/LSTM experiments)
templates/        Jinja2 HTML templates
static/           CSS, JS and images for the UI
```

## Running locally

Requirements: Python 3.10+, MySQL 8.

```bash
git clone https://github.com/<your-username>/autism-screening-ml.git
cd autism-screening-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

mysql -u root -p < db.sql          # creates the `asd` database and `users` table

cp .env.example .env               # then edit the values
export $(cat .env | xargs)          # Windows: set each variable manually
python app.py                      # open http://127.0.0.1:5000
```

## Tech stack

Python · Flask · MySQL · scikit-learn · XGBoost · pandas · TensorFlow/Keras (notebook) · HTML/CSS/Bootstrap

## Credits

- Dataset: Fadi Thabtah, UCI Machine Learning Repository (CC BY 4.0)
- Hero photo: RDNE Stock Project on [Pexels](https://www.pexels.com/)
