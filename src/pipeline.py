import argparse
from pathlib import Path

import pandas as pd

from predict_m1 import load_m1, predict_m1
from predict_m2 import load_m2, predict_m2


def read_table(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".tsv":
        return pd.read_csv(path_obj, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path_obj)

    raise ValueError("지원하지 않는 파일 형식입니다. .csv 또는 .tsv 파일을 사용하세요.")


def write_table(df: pd.DataFrame, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    suffix = path_obj.suffix.lower()
    if suffix == ".tsv":
        df.to_csv(path_obj, sep="\t", index=False)
    else:
        df.to_csv(path_obj, index=False)


def predict_pipeline(
    text: str,
    m1_tokenizer,
    m1_model,
    m1_device: str,
    m2_tokenizer,
    m2_model,
    m2_device: str,
    m1_threshold: float | None = None,
):
    """
    M1 → M2 파이프라인 단일 예측.

    흐름:
      1. M1으로 INFORMATIVE / UNINFORMATIVE 판단
      2. M1 결과가 INFORMATIVE일 때만 M2 실행
      3. 최종 결과 반환
    """
    m1_result = predict_m1(
        text,
        tokenizer=m1_tokenizer,
        model=m1_model,
        device=m1_device,
        informative_threshold=m1_threshold,
    )

    if m1_result["label"] == "INFORMATIVE":
        m2_result = predict_m2(
            text,
            tokenizer=m2_tokenizer,
            model=m2_model,
            device=m2_device,
        )

        final_label = f"INFORMATIVE / {m2_result['label']}"
        return {
            "Text": text,
            "m1_label": m1_result["label"],
            "m1_confidence": round(m1_result["confidence"], 6),
            "m2_label": m2_result["label"],
            "m2_confidence": round(m2_result["confidence"], 6),
            "final_label": final_label,
            "m1_probabilities": m1_result["probabilities"],
            "m2_probabilities": m2_result["probabilities"],
        }

    return {
        "Text": text,
        "m1_label": m1_result["label"],
        "m1_confidence": round(m1_result["confidence"], 6),
        "m2_label": "-",
        "m2_confidence": "-",
        "final_label": "UNINFORMATIVE",
        "m1_probabilities": m1_result["probabilities"],
        "m2_probabilities": "-",
    }


def run_file(
    input_path: str,
    output_path: str,
    text_column: str,
    m1_model_dir: str,
    m2_model_dir: str,
    m1_threshold: float | None = None,
    limit: int | None = None,
):
    df = read_table(input_path)

    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")

    if limit is not None:
        df = df.head(limit).copy()

    print("Loading M1 model...")
    m1_tokenizer, m1_model, m1_device = load_m1(m1_model_dir)

    print("Loading M2 model...")
    m2_tokenizer, m2_model, m2_device = load_m2(m2_model_dir)

    rows = []
    for text in df[text_column].astype(str).tolist():
        rows.append(
            predict_pipeline(
                text,
                m1_tokenizer=m1_tokenizer,
                m1_model=m1_model,
                m1_device=m1_device,
                m2_tokenizer=m2_tokenizer,
                m2_model=m2_model,
                m2_device=m2_device,
                m1_threshold=m1_threshold,
            )
        )

    result_df = pd.DataFrame(rows)
    write_table(result_df, output_path)
    print(f"Saved pipeline result to: {output_path}")
    return result_df


def main():
    parser = argparse.ArgumentParser(description="M1 informative classifier + M2 risk classifier pipeline")

    parser.add_argument("--text", type=str, default=None, help="단일 tweet 입력")
    parser.add_argument("--input_path", type=str, default=None, help="CSV/TSV 입력 파일 경로")
    parser.add_argument("--output_path", type=str, default="outputs/pipeline_result.csv", help="결과 저장 경로")
    parser.add_argument("--text_column", type=str, default="Text", help="입력 파일에서 tweet text 컬럼명")

    parser.add_argument("--m1_model_dir", type=str, default="models/m1_bertweet_covid_freeze_final")
    parser.add_argument("--m2_model_dir", type=str, default="models/m2_distilbert_risk")
    parser.add_argument("--m1_threshold", type=float, default=None, help="P(INFORMATIVE) threshold. 기본값은 argmax")
    parser.add_argument("--limit", type=int, default=None, help="파일 입력 시 앞에서 N개만 실행")

    args = parser.parse_args()

    if args.text is None and args.input_path is None:
        raise ValueError("--text 또는 --input_path 중 하나는 반드시 입력해야 합니다.")

    if args.text is not None:
        m1_tokenizer, m1_model, m1_device = load_m1(args.m1_model_dir)
        m2_tokenizer, m2_model, m2_device = load_m2(args.m2_model_dir)

        result = predict_pipeline(
            args.text,
            m1_tokenizer=m1_tokenizer,
            m1_model=m1_model,
            m1_device=m1_device,
            m2_tokenizer=m2_tokenizer,
            m2_model=m2_model,
            m2_device=m2_device,
            m1_threshold=args.m1_threshold,
        )

        for key, value in result.items():
            print(f"{key}: {value}")

    else:
        result_df = run_file(
            input_path=args.input_path,
            output_path=args.output_path,
            text_column=args.text_column,
            m1_model_dir=args.m1_model_dir,
            m2_model_dir=args.m2_model_dir,
            m1_threshold=args.m1_threshold,
            limit=args.limit,
        )
        print(result_df.head())


if __name__ == "__main__":
    main()
