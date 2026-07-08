# -4-
SNS 게시글에서 단순 감정 표현과 실제 현장 이상 징후를 구분해 감염병 확산 위험 신호를 탐지하고 알리는 AI 시스템 제작 (키워드 기반 분류 -> 간단한 NLP 모델을 활용해서 위험도 감지, 경고 구현) 

# COVID-19 SNS 위험 신호 탐지 파이프라인

SNS 게시글에서 코로나19 관련 정보성 게시글을 선별하고, 정보성 게시글에 대해 감염병 확산 위험도를 `Low`, `Medium`, `High`로 분류하는 2단계 AI 파이프라인입니다.

본 프로젝트는 단순 감정 표현이나 잡담성 게시글과 실제 감염병 관련 정보가 포함된 게시글을 구분한 뒤, 정보성 게시글만 대상으로 위험도를 판단하는 구조로 구현되었습니다.

## 프로젝트 개요

### 목표

- 코로나19 관련 SNS 게시글이 실제 정보성 게시글인지 판별
- 정보성 게시글에 대해서만 위험도 분류 수행
- 최종적으로 게시글별 감염병 확산 위험 신호를 자동 탐지

### 전체 구조

```text
Input Tweet
   ↓
M1: Informative / Uninformative 분류
   ↓
INFORMATIVE인 경우에만 M2 실행
   ↓
M2: Low / Medium / High 위험도 분류
   ↓
Final Output
```

## 모델 구성

### M1: 정보성 게시글 분류 모델

M1은 입력 tweet이 코로나19 관련 정보성 게시글인지 아닌지를 분류합니다.

- 모델 역할: `INFORMATIVE` / `UNINFORMATIVE` 이진 분류
- 기반 모델: `vinai/bertweet-covid19-base-cased`
- 출력 라벨:
  - `INFORMATIVE`
  - `UNINFORMATIVE`

M1 추론 시에는 학습에 사용한 원본 tokenizer를 명시적으로 사용합니다.

```text
M1 tokenizer: vinai/bertweet-covid19-base-cased
M1 model weights: models/m1_final_verified
```

### M2: 위험도 분류 모델

M2는 M1에서 `INFORMATIVE`로 판단된 게시글에 대해서만 실행됩니다.

- 모델 역할: 정보성 게시글의 위험도 분류
- 기반 모델: DistilBERT fine-tuning 모델
- 출력 라벨:
  - `Low`
  - `Medium`
  - `High`

M1이 `UNINFORMATIVE`로 판단한 경우에는 M2를 실행하지 않고 최종 결과를 `UNINFORMATIVE`로 반환합니다.

## 디렉토리 구조

프로젝트 실행을 위해 다음과 같은 구조를 권장합니다.

```text
project-root/
├─ data/
│  └─ demo/
│     └─ pipeline_demo_input.tsv
├─ models/
│  ├─ m1_final_verified/
│  └─ m2_distilbert_risk/
├─ outputs/
├─ src/
│  ├─ predict_m1.py
│  ├─ predict_m2.py
│  └─ pipeline.py
├─ requirements.txt
└─ README.md
```

### 모델 폴더

모델 폴더는 기본적으로 다음 위치에 두어야 합니다.

```text
models/
├─ m1_final_verified/
└─ m2_distilbert_risk/
```

각 모델 폴더 안에는 Hugging Face 형식의 모델 파일이 포함되어야 합니다.

예시:

```text
models/m1_final_verified/
├─ config.json
├─ model.safetensors 또는 pytorch_model.bin
└─ 기타 모델 관련 파일
```

모델 파일의 용량이 큰 경우 GitHub에 직접 포함하지 않고 Google Drive, Hugging Face Hub, Git LFS 등을 사용할 수 있습니다.

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. 가상환경 생성 및 활성화

Windows PowerShell 기준:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux 기준:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 패키지 설치

`requirements.txt`가 있는 경우:

```bash
pip install -r requirements.txt
```

`requirements.txt`가 없거나 수동 설치가 필요한 경우 다음 패키지를 설치합니다.

```bash
pip install torch transformers pandas numpy scikit-learn sentencepiece protobuf emoji==0.6.0
```

BERTweet 계열 tokenizer 사용을 위해 `emoji==0.6.0` 설치가 필요할 수 있습니다.

## 실행 방법

### 1. 단일 문장 추론

하나의 raw text를 직접 입력해 파이프라인을 실행할 수 있습니다.

```bash
python src/pipeline.py --text "A patient in South Korea who was the 31st confirmed case infected many people." --m1_model_dir models/m1_final_verified --m2_model_dir models/m2_distilbert_risk
```

실행 결과 예시:

```text
Text: A patient in South Korea who was the 31st confirmed case infected many people.
m1_label: INFORMATIVE
m1_confidence: ...
m2_label: High
m2_confidence: ...
final_label: INFORMATIVE / High
```

### 2. 배치 파일 추론

TSV 또는 CSV 파일을 입력으로 넣어 여러 문장을 한 번에 추론할 수 있습니다.

```bash
python src/pipeline.py --input_path data/demo/pipeline_demo_input.tsv --text_column Text --output_path outputs/pipeline_result_full.csv --m1_model_dir models/m1_final_verified --m2_model_dir models/m2_distilbert_risk
```

실행이 완료되면 `outputs/pipeline_result_full.csv`에 결과가 저장됩니다.

### 3. 결과 분포 확인

파이프라인 결과 파일에서 최종 라벨 분포와 M1 통과율을 확인할 수 있습니다.

```bash
python -c "import pandas as pd; df=pd.read_csv('outputs/pipeline_result_full.csv'); print(df[['m1_label','final_label']].value_counts()); print('pass rate:', (df['m1_label']=='INFORMATIVE').mean())"
```

## 출력 컬럼

파이프라인 결과 파일에는 다음 컬럼이 포함됩니다.

```text
Text
m1_label
m1_confidence
m2_label
m2_confidence
final_label
m1_probabilities
m2_probabilities
```

각 컬럼의 의미는 다음과 같습니다.

| 컬럼명 | 설명 |
|---|---|
| `Text` | 입력 게시글 원문 |
| `m1_label` | M1의 정보성 분류 결과 |
| `m1_confidence` | M1 예측 confidence |
| `m2_label` | M2의 위험도 분류 결과 |
| `m2_confidence` | M2 예측 confidence |
| `final_label` | 최종 파이프라인 결과 |
| `m1_probabilities` | M1 클래스별 확률 |
| `m2_probabilities` | M2 클래스별 확률 |

M1이 `UNINFORMATIVE`로 예측한 경우 M2는 실행되지 않으며, M2 관련 컬럼은 `-`로 표시됩니다.

## 최종 파이프라인 테스트 결과

최종 파이프라인은 사전에 구성한 데모 입력 파일 `data/demo/pipeline_demo_input.tsv`를 이용해 테스트하였다.

실행 명령어는 다음과 같다.

```bash
python src/pipeline.py --input_path data/demo/pipeline_demo_input.tsv --text_column Text --output_path outputs/pipeline_result_full.csv --m1_model_dir models/m1_final_verified --m2_model_dir models/m2_distilbert_risk

테스트 결과, 총 20개의 데모 입력 문장 중 16개가 M1에서 INFORMATIVE로 분류되어 M2 위험도 분류 단계까지 전달되었고, 4개는 UNINFORMATIVE로 분류되어 M2를 실행하지 않았다.

Total samples: 20
M1 pass to M2: 16
M1 filtered as UNINFORMATIVE: 4
M1 pass rate: 80.00%

최종 출력 분포는 다음과 같다.

INFORMATIVE / High      9
INFORMATIVE / Low       5
INFORMATIVE / Medium    2
UNINFORMATIVE           4

이를 통해 M1이 정보성 게시글 여부를 먼저 판단하고, 정보성 게시글에 대해서만 M2가 위험도(Low, Medium, High)를 분류하는 end-to-end 파이프라인이 정상적으로 동작함을 확인하였다.

## 구현상 주의사항

M1은 BERTweet-COVID19 기반 모델로 학습되었기 때문에, 추론 시 tokenizer를 반드시 다음과 같이 불러옵니다.

```python
tokenizer = AutoTokenizer.from_pretrained(
    "vinai/bertweet-covid19-base-cased",
    use_fast=False
)
```

모델 weight는 fine-tuned checkpoint에서 불러옵니다.

```python
model = AutoModelForSequenceClassification.from_pretrained("models/m1_final_verified")
```

즉, M1 추론 구조는 다음과 같습니다.

```text
tokenizer → vinai/bertweet-covid19-base-cased
model     → models/m1_final_verified
```

이 구조를 유지해야 학습 시점과 동일한 tokenization으로 raw text 추론을 수행할 수 있습니다.

## 문제 해결

### 1. 모델 경로 오류

다음과 같은 오류가 발생하면 모델 폴더 경로가 잘못되었을 가능성이 있습니다.

```text
Repository Not Found
is not a local folder and is not a valid model identifier
```

모델 폴더가 다음 위치에 있는지 확인합니다.

```text
models/m1_final_verified/
models/m2_distilbert_risk/
```

또한 압축 해제 시 폴더가 중첩되지 않도록 주의해야 합니다.

잘못된 예:

```text
models/m1_final_verified/m1_final_verified/config.json
```

올바른 예:

```text
models/m1_final_verified/config.json
```

### 2. M1이 모든 입력을 UNINFORMATIVE로 예측하는 경우

M1 tokenizer를 저장된 checkpoint 폴더에서 불러오고 있지 않은지 확인합니다.

올바른 방식:

```python
AutoTokenizer.from_pretrained("vinai/bertweet-covid19-base-cased", use_fast=False)
```

모델 weight는 local checkpoint에서 불러오되, tokenizer는 원본 base tokenizer를 사용해야 합니다.

### 3. 결과 파일이 생성되지 않는 경우

`outputs/` 폴더가 없으면 직접 생성합니다.

```bash

mkdir outputs
```

그 후 batch inference 명령어를 다시 실행합니다.

## 사용 데이터

본 프로젝트는 코로나19 관련 tweet 데이터를 기반으로 합니다.

- M1 데이터: `INFORMATIVE` / `UNINFORMATIVE` 라벨이 포함된 COVID-19 tweet 데이터
- M2 데이터: M1 데이터 중 `INFORMATIVE` 샘플에 대해 위험도 `Low`, `Medium`, `High`를 추가 라벨링한 데이터

M2는 M1에서 정보성으로 판단된 tweet에 대해서만 실행되도록 설계되었습니다.

## 프로젝트 한계

- M1이 실제 정보성 게시글을 `UNINFORMATIVE`로 잘못 예측하면 M2가 실행되지 않습니다.
- 따라서 전체 파이프라인 성능은 M1의 informative recall에 영향을 받습니다.
- M2는 M1을 통과한 게시글에 대해서만 위험도를 분류합니다.


## 라이선스 및 참고

본 프로젝트는 AI 학회 미니 프로젝트 제출용으로 작성되었습니다.

