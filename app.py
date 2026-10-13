"""Flask web app for autism screening predictions from AQ-10 questionnaire answers."""
import os
from functools import wraps

import mysql.connector
from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import ml

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME", "asd"),
}
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "autism_child_data.csv")

# Dataset used by the upload -> split -> model workflow (defaults to the bundled data).
state = {"df": ml.prepare(ml.load_raw(DATA_PATH)), "split": None}

# Model used for predictions: linear SVM trained once at startup on the bundled data.
predictor = ml.train_final(state["df"], "SVM")


def db_query(query, values=(), fetch=False):
    conn = mysql.connector.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(query, values)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        conn.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if password != request.form["c_password"]:
            return render_template("register.html", message="Passwords do not match!")
        if db_query("SELECT id FROM users WHERE email = %s", (email,), fetch=True):
            return render_template("register.html", message="This email ID already exists!")
        db_query(
            "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
            (name, email, generate_password_hash(password)),
        )
        return render_template("login.html", message="Successfully registered!")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        rows = db_query("SELECT password_hash FROM users WHERE email = %s", (email,), fetch=True)
        if not rows:
            return render_template("login.html", message="This email ID does not exist!")
        if not check_password_hash(rows[0][0], request.form["password"]):
            return render_template("login.html", message="Invalid password!")
        session["user_email"] = email
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files["file"]
        try:
            if file.filename.lower().endswith((".xlsx", ".xls")):
                raw = ml.pd.read_excel(file)
            else:
                raw = ml.pd.read_csv(file)
            state["df"] = ml.prepare(raw)
        except (KeyError, ValueError) as err:
            return render_template("upload.html", message=f"Could not read this dataset: {err}")
        state["split"] = None
        return render_template(
            "upload.html",
            data=raw.head(100).to_html(),
            message=f"Dataset uploaded: {len(state['df'])} unique records after removing duplicates. Go to Data split!",
        )
    return render_template("upload.html")


@app.route("/split", methods=["GET", "POST"])
@login_required
def split():
    if request.method == "POST":
        test_size = float(request.form["split_size"])
        if not 0.05 <= test_size <= 0.5:
            return render_template("split.html", message="Test size must be between 0.05 and 0.5.")
        state["split"] = ml.split(state["df"], test_size)
        return render_template("model.html", message="Data split successfully! Choose a model.")
    return render_template("split.html")


@app.route("/model", methods=["GET", "POST"])
@login_required
def model():
    if request.method == "POST":
        algorithm = request.form["algorithm"]
        if algorithm not in ml.ALGORITHMS:
            return render_template("model.html", message="Algorithm not recognized!")
        if state["split"] is None:
            state["split"] = ml.split(state["df"])
        holdout = ml.holdout_scores(algorithm, *state["split"])
        cv = ml.cv_scores(algorithm, state["df"])
        return render_template(
            "model.html",
            algorithm=algorithm,
            accuracy=f"{holdout['accuracy'] * 100:.1f}",
            cv_accuracy=f"{cv['accuracy'] * 100:.1f}",
            cv_std=f"{cv['accuracy_std'] * 100:.1f}",
            message="Accuracy computed on the held-out test split and with 5-fold cross-validation.",
        )
    return render_template("model.html")


@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    if request.method == "POST":
        try:
            answers = [int(request.form[f"a{i}"]) for i in range(1, 11)]
            age = float(request.form["age"])
            flags = [int(request.form[k]) for k in ("gender", "jundice", "autism")]
        except (KeyError, ValueError):
            return render_template("prediction.html", prediction="Please fill in every field.")
        if any(v not in (0, 1) for v in answers + flags) or not 1 <= age <= 18:
            return render_template("prediction.html", prediction="Please fill in every field.")

        features = ml.pd.DataFrame([answers + [age] + flags], columns=ml.FEATURE_COLS)
        if predictor.predict(features)[0] == 1:
            result = "Screening indicates ASD traits. Please consult a specialist."
        else:
            result = "Screening does not indicate ASD traits."
        return render_template("prediction.html", prediction=result)
    return render_template("prediction.html")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
