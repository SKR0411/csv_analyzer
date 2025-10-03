# app.py
from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)

df_global = None  # store uploaded dataframe globally

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/topics")
def topics():
    topics_list = [
        {"title": "Statistics", "desc": "The foundation of data science, used to analyze and summarize data."},
        {"title": "Machine Learning", "desc": "Algorithms that learn patterns from data and make predictions."},
        {"title": "Data Visualization", "desc": "Techniques to present data with graphs and charts."},
        {"title": "Deep Learning", "desc": "Neural networks for image recognition, NLP, and AI tasks."},
        {"title": "Big Data", "desc": "Handling very large datasets with tools like Hadoop & Spark."}
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
            summary = df.describe().to_html(classes="table table-striped")
            head = df.head().to_html(classes="table table-bordered")
            return render_template("analyze.html", summary=summary, head=head)
    return render_template("upload.html")

@app.route("/download_summary")
def download_summary():
    global df_global
    if df_global is not None:
        summary_csv = df_global.describe().to_csv()
        return send_file(
            io.BytesIO(summary_csv.encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name="summary.csv"
        )
    return "No data uploaded yet."

@app.route("/plot/<plot_type>")
def plot(plot_type):
    global df_global
    if df_global is None:
        return "No data uploaded yet. Please upload a CSV first."

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