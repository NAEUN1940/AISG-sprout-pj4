import os
import re
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# 글로벌 변수
vectorizer = None
info_model = None

CATEGORIES = {
    "confirmed": [r"\bconfirmed\b", r"\bconfirm(?:s|ed|ing)?\b", r"\bpositive\b", r"\btested positive\b", r"\bnew cases?\b", r"\bnew infections?\b", r"\bcase(?:s)?\b"],
    "suspected": [r"\bsuspected\b", r"\bsuspect(?:s|ed|ing)?\b", r"\bsymptoms?\b", r"\bsymptomatic\b", r"\bunder observation\b", r"\bquarantine(?:d)?\b", r"\bisolated\b"],
    "death": [r"\bdeaths?\b", r"\bdied\b", r"\bdies\b", r"\bfatalit(?:y|ies)\b", r"\bdeceased\b", r"\bpassed away\b"],
    "recovered": [r"\brecovered\b", r"\brecover(?:s|ed|ing|ies)?\b", r"\bdischarged\b", r"\bdischarge(?:s|d)?\b"],
    "tested": [r"\btested\b", r"\btesting\b", r"\btests?\b", r"\bscreened\b", r"\bscreening\b", r"\bswab\b"],
}

def classify_info_category(text):
    text = str(text).lower()
    matched = []
    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.append(category)
                break
    return matched

def init_model():
    global vectorizer, info_model
    train_path = "train.tsv"
    if not os.path.exists(train_path):
        print(f"❌ 에러: {train_path} 파일이 없습니다.")
        return
    print("🤖 모델 학습 중...")
    train_df = pd.read_csv(train_path, sep="\t")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
    X_train_vec = vectorizer.fit_transform(train_df["Text"].astype(str))
    info_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    info_model.fit(X_train_vec, train_df["Label"].astype(str))
    print("✅ 학습 완료!")

@app.route("/")
def index():
    return render_template("index.html")

# 오류가 났던 72번 줄 수정 완료!
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files: return jsonify({"error": "파일 없음"}), 400
    file = request.files["file"]
    try:
        df = pd.read_csv(file, sep="\t", header=None)
        if df.iloc[0, 1].lower() == "text": df = df.iloc[1:].reset_index(drop=True)
        df.columns = ["Id", "Text", "Label"]
        X_vec = vectorizer.transform(df["Text"].astype(str))
        preds = info_model.predict(X_vec)
        results = []
        for idx, row in df.iterrows():
            if preds[idx] == "INFORMATIVE":
                results.append({"id": str(row["Id"]), "text": str(row["Text"]), "categories": classify_info_category(row["Text"])})
        return jsonify({"tweets": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_model()
    app.run(debug=True, port=5000)