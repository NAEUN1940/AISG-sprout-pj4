
import os
import re
import io
import json
import pandas as pd

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

DATA_DIR = os.environ.get("WNUT_DATA_DIR", "data/WNUT-2020-Task-2-Dataset")

CATEGORIES = {
    "confirmed": [
        r"\bconfirmed\b",
        r"\bconfirm(?:s|ed|ing)?\b",
        r"\bpositive\b",
        r"\btested positive\b",
        r"\bnew cases?\b",
        r"\bnew infections?\b",
        r"\bcase(?:s)?\b",
    ],
    "suspected": [
        r"\bsuspected\b",
        r"\bsuspect(?:s|ed|ing)?\b",
        r"\bsymptoms?\b",
        r"\bsymptomatic\b",
        r"\bunder observation\b",
        r"\bquarantine(?:d)?\b",
        r"\bisolated\b",
    ],
    "death": [
        r"\bdeaths?\b",
        r"\bdied\b",
        r"\bdies\b",
        r"\bfatalit(?:y|ies)\b",
        r"\bdeceased\b",
        r"\bpassed away\b",
    ],
    "recovered": [
        r"\brecovered\b",
        r"\brecover(?:s|ed|ing|ies)?\b",
        r"\bdischarged\b",
        r"\bdischarge(?:s|d)?\b",
    ],
    "tested": [
        r"\btested\b",
        r"\btesting\b",
        r"\btests?\b",
        r"\bscreened\b",
        r"\bscreening\b",
        r"\bswab\b",
    ],
}

CATEGORY_ORDER = ["confirmed", "suspected", "death", "recovered", "tested", "other_informative", "NONINFORMATIVE"]

vectorizer = None
info_model = None
model_status = {
    "ready": False,
    "message": "모델이 아직 로드되지 않았습니다."
}


def classify_info_category(text):
    text = str(text).lower()
    matched = []

    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.append(category)
                break

    if matched:
        return ";".join(matched)
    return "other_informative"


def normalize_input_df(df):
    """
    업로드 TSV가 아래 중 어떤 형태여도 최대한 자동 인식:
    1) Id, Text, Label 헤더 있음
    2) id, text, label처럼 소문자
    3) 헤더 없이 3열: Id / Text / Label
    4) 헤더 없이 2열: Id / Text
    5) 헤더 없이 1열: Text
    """
    df = df.copy()

    lower_map = {str(c).strip().lower(): c for c in df.columns}
    text_col = None
    for candidate in ["text", "sentence", "tweet", "content", "info"]:
        if candidate in lower_map:
            text_col = lower_map[candidate]
            break

    # 헤더 없이 읽혔거나 컬럼명이 이상한 경우 처리
    if text_col is None:
        cols = list(df.columns)
        if len(cols) >= 3:
            # WNUT valid/test 형식은 보통 0, 1, 2 = Id, Text, Label
            df = df.rename(columns={cols[0]: "Id", cols[1]: "Text", cols[2]: "Label"})
        elif len(cols) == 2:
            df = df.rename(columns={cols[0]: "Id", cols[1]: "Text"})
            df["Label"] = ""
        elif len(cols) == 1:
            df = df.rename(columns={cols[0]: "Text"})
            df["Id"] = range(1, len(df) + 1)
            df["Label"] = ""
        else:
            raise ValueError("TSV에서 문장 컬럼을 찾을 수 없습니다.")
    else:
        df = df.rename(columns={text_col: "Text"})
        if "id" in lower_map and lower_map["id"] in df.columns:
            df = df.rename(columns={lower_map["id"]: "Id"})
        if "label" in lower_map and lower_map["label"] in df.columns:
            df = df.rename(columns={lower_map["label"]: "Label"})
        if "Id" not in df.columns:
            df["Id"] = range(1, len(df) + 1)
        if "Label" not in df.columns:
            df["Label"] = ""

    return df[["Id", "Text", "Label"]].fillna("")


def try_train_model():
    global vectorizer, info_model, model_status

    train_path = os.path.join(DATA_DIR, "train.tsv")
    if not os.path.exists(train_path):
        model_status = {
            "ready": False,
            "message": f"train.tsv를 찾지 못했습니다. 현재 경로: {train_path}"
        }
        return

    train_df = pd.read_csv(train_path, sep="\t")
    if "Text" not in train_df.columns or "Label" not in train_df.columns:
        model_status = {
            "ready": False,
            "message": "train.tsv에 Text, Label 컬럼이 필요합니다."
        }
        return

    X_train = train_df["Text"].astype(str)
    y_train = train_df["Label"].astype(str)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    info_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    info_model.fit(X_train_vec, y_train)

    model_status = {
        "ready": True,
        "message": f"모델 로드 완료: {train_path}"
    }


try_train_model()


@app.route("/")
def index():
    return render_template("index.html", model_status=model_status)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "TSV 파일이 업로드되지 않았습니다."}), 400

    file = request.files["file"]
    raw = file.read()
    text = raw.decode("utf-8-sig", errors="replace")

    # 사용자가 업로드한 TSV가 헤더가 있는지 없는지 자동 추정
    first_line = text.splitlines()[0] if text.splitlines() else ""
    maybe_header = any(x.lower() in first_line.lower().split("\t") for x in ["text", "sentence", "tweet", "content", "info"])

    if maybe_header:
        df = pd.read_csv(io.StringIO(text), sep="\t")
    else:
        df = pd.read_csv(io.StringIO(text), sep="\t", header=None)

    df = normalize_input_df(df)

    if not model_status["ready"]:
        return jsonify({
            "error": "LogisticRegression 모델이 준비되지 않았습니다.",
            "detail": model_status["message"],
            "hint": "ipynb와 같은 방식으로 돌리려면 data/WNUT-2020-Task-2-Dataset/train.tsv 파일이 필요합니다."
        }), 500

    X = df["Text"].astype(str)
    X_vec = vectorizer.transform(X)

    result = df.copy()
    result["pred_label"] = info_model.predict(X_vec)
    result["pred_info_categories"] = "NONINFORMATIVE"

    info_mask = result["pred_label"] == "INFORMATIVE"
    result.loc[info_mask, "pred_info_categories"] = result.loc[info_mask, "Text"].apply(classify_info_category)

    result["split"] = "uploaded_tsv"

    rows = result.to_dict(orient="records")

    counts = {c: 0 for c in CATEGORY_ORDER}
    info_count = int((result["pred_label"] == "INFORMATIVE").sum())
    non_count = int((result["pred_label"] == "NONINFORMATIVE").sum())

    for cats in result["pred_info_categories"].astype(str):
        for c in cats.split(";"):
            if c:
                counts[c] = counts.get(c, 0) + 1

    return jsonify({
        "rows": rows,
        "summary": {
            "total": len(result),
            "informative": info_count,
            "noninformative": non_count,
            "counts": counts
        },
        "model_status": model_status
    })


@app.route("/health")
def health():
    return jsonify(model_status)


if __name__ == "__main__":
    app.run(debug=True)
