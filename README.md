
# TSV Informative + 5 Category Classifier Web App

이 웹앱은 사용자가 올린 Jupyter notebook의 흐름을 그대로 웹에서 돌리도록 만든 버전입니다.

## 하는 일

1. TSV 파일 업로드
2. `TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2)`로 문장 벡터화
3. `LogisticRegression(max_iter=1000, class_weight="balanced")`로 INFORMATIVE / NONINFORMATIVE 예측
4. INFORMATIVE 문장만 키워드셋 기반으로 5개 카테고리 분류
   - confirmed
   - suspected
   - death
   - recovered
   - tested
   - 키워드에 안 걸리면 other_informative

## 폴더 구조

아래처럼 WNUT 데이터셋을 넣어야 모델이 학습됩니다.

```text
info_classifier_webapp/
  app.py
  requirements.txt
  templates/
    index.html
  data/
    WNUT-2020-Task-2-Dataset/
      train.tsv
      valid.tsv
      test.tsv
```

중요: `train.tsv`에는 `Text`, `Label` 컬럼이 있어야 합니다.

## 실행 방법

```bash
pip install -r requirements.txt
python app.py
```

그 다음 브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:5000
```

## 업로드 TSV 형식

아래 형식을 자동으로 인식합니다.

### 헤더 있는 경우

```tsv
Id	Text	Label
1	Two new confirmed cases were reported	INFORMATIVE
2	Stay safe everyone	NONINFORMATIVE
```

또는 Text 컬럼만 있어도 됩니다.

```tsv
Text
Two new confirmed cases were reported
Stay safe everyone
```

### 헤더 없는 경우

```tsv
1	Two new confirmed cases were reported	INFORMATIVE
2	Stay safe everyone	NONINFORMATIVE
```
