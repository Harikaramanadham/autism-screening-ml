"""Data preparation, model definitions and evaluation for the autism screening app.

Dataset: UCI "Autistic Spectrum Disorder Screening Data for Children" (ages 4-11),
AQ-10 questionnaire answers plus a few demographic fields.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

QUESTION_COLS = [f"A{i}_Score" for i in range(1, 11)]
FEATURE_COLS = QUESTION_COLS + ["age", "gender", "jaundice", "family_autism"]
TARGET_COL = "asd"


def load_raw(path):
    """Read the dataset from a .csv or .xlsx file."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def prepare(raw: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean the raw UCI table into numeric features + binary target.

    The raw file contains many exact duplicate rows (about half of them), which
    leak between train and test splits and inflate accuracy, so they are
    dropped by default.
    """
    df = raw.copy()
    if drop_duplicates:
        df = df.drop_duplicates()
    df = df[df["age"].astype(str) != "?"]

    out = df[QUESTION_COLS].astype(int).copy()
    out["age"] = pd.to_numeric(df["age"])
    out["gender"] = (df["gender"] == "m").astype(int)            # 1 = male
    out["jaundice"] = (df["jundice"] == "yes").astype(int)       # born with jaundice
    out["family_autism"] = (df["austim"] == "yes").astype(int)   # family member with ASD
    out[TARGET_COL] = (df["Class/ASD"] == "YES").astype(int)
    return out.reset_index(drop=True)


def make_model(name: str):
    """Return an unfitted model for the given algorithm name."""
    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=42)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=500, random_state=0),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=0),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42,
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown algorithm: {name}")
    return models[name]


ALGORITHMS = ["SVM", "Logistic Regression", "Random Forest Classifier", "Decision Tree Classifier", "XGBoost"]


def split(df: pd.DataFrame, test_size: float = 0.3):
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def holdout_scores(name, X_train, X_test, y_train, y_test):
    """Fit on the training split and score on the held-out test split."""
    model = make_model(name).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)}


def cv_scores(name, df: pd.DataFrame, folds: int = 5):
    """Stratified k-fold cross-validation on the full (deduplicated) data."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    r = cross_validate(make_model(name), df[FEATURE_COLS], df[TARGET_COL], cv=cv, scoring=["accuracy", "f1"])
    return {
        "accuracy": r["test_accuracy"].mean(),
        "accuracy_std": r["test_accuracy"].std(),
        "f1": r["test_f1"].mean(),
    }


def train_final(df: pd.DataFrame, name: str = "SVM"):
    """Fit the deployed model on all available data."""
    return make_model(name).fit(df[FEATURE_COLS], df[TARGET_COL])
