# app.py
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/topics")
def topics():
    topics_list = [
        {"title": "Statistics", "desc": "The foundation of data science, used to analyze and summarize data."},
        {"title": "Machine Learning", "desc": "Algorithms that learn patterns from data and make predictions."},
        {"title": "Data Visualization", "desc": "Techniques to present data with graphs and charts."}
    ]
    return render_template("topics.html", topics=topics_list)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            summary = df.describe().to_html(classes="table table-striped")
            return render_template("analyze.html", summary=summary)
    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)