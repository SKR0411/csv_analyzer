from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import io

app = Flask(__name__)
df_global = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/topics")
def topics():
    topics_list = [
        {"title": "Statistics", "desc": "Analyzing and summarizing data."},
        {"title": "Machine Learning", "desc": "Algorithms that learn patterns and predict outcomes."},
        {"title": "Data Visualization", "desc": "Charts and plots to explore insights."},
        {"title": "Deep Learning", "desc": "Neural networks for advanced AI tasks."},
        {"title": "Big Data", "desc": "Handling massive datasets using distributed tools."},
        {"title": "Natural Language Processing", "desc": "Analyzing and understanding human language."},
        {"title": "Data Cleaning", "desc": "Handling missing values, duplicates, and noisy data."}
    ]
    return render_template("topics.html", topics=topics_list)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    global df_global
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            df_global = df
            summary = df.describe(include="all").to_html(classes="table table-striped")
            head = df.head().to_html(classes="table table-bordered")
            dtypes = df.dtypes.to_frame("dtype").to_html(classes="table table-sm table-hover")
            missing = df.isnull().sum().to_frame("missing_values").to_html(classes="table table-sm table-hover")
            return render_template("analyze.html", summary=summary, head=head, dtypes=dtypes, missing=missing, columns=df.columns)
    return render_template("upload.html")

@app.route("/plot/<plot_type>")
def plot(plot_type):
    global df_global
    if df_global is None:
        return "No data uploaded yet."

    img = io.BytesIO()
    plt.figure(figsize=(6,4))

    if plot_type == "hist":
        df_global.hist(figsize=(8,6))
        plt.tight_layout()
    elif plot_type == "corr":
        sns.heatmap(df_global.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")

    plt.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")

@app.route("/scatter", methods=["POST"])
def scatter():
    global df_global
    xcol = request.form["xcol"]
    ycol = request.form["ycol"]

    img = io.BytesIO()
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df_global[xcol], y=df_global[ycol])
    plt.title(f"Scatter Plot: {xcol} vs {ycol}")
    plt.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")

@app.route("/ml", methods=["POST"])
def ml_demo():
    global df_global
    target = request.form["target"]

    X = df_global.drop(columns=[target]).select_dtypes(include="number")
    y = df_global[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    img = io.BytesIO()
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, preds, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Linear Regression (R² = {score:.2f})")
    plt.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")
    
if __name__ == "__main__":
    app.run(debug=True)