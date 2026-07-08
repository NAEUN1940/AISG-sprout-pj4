# COVID-19 Tweet 위험도 분류 모델 (M2)

## 1. 프로젝트 개요

본 프로젝트는 COVID-19 관련 트윗을 분석하여 정보성 여부와 위험도를 분류하는 파이프라인을 구축하는 것을 목표로 합니다.

전체 파이프라인은 다음과 같이 구성됩니다.

```text
Tweet
→ M1: Informative / Non-informative 분류
→ Informative로 분류된 tweet만 M2로 전달
→ M2: Low / Medium / High 위험도 분류
```

이 README는 그중 **M2 위험도 분류 모델**에 대한 설명입니다.

---

## 2. M2 Task 정의

M2 모델은 M1에서 informative로 분류된 COVID-19 관련 트윗을 입력으로 받아, 해당 트윗의 위험도를 다음 세 가지 라벨 중 하나로 분류합니다.

| Label | 의미 |
|---|---|
| Low | 상대적으로 낮은 위험도 |
| Medium | 중간 수준의 위험도 |
| High | 높은 위험도 |

---

## 3. 데이터셋

M2 위험도 분류에는 **informative tweet만 사용**했습니다.

원본 데이터에서 중복 tweet을 제거한 뒤 최종적으로 사용한 데이터는 총 **4,685개**입니다.

### 라벨 분포

| Risk Level | Count |
|---|---:|
| Low | 823 |
| Medium | 1,181 |
| High | 2,681 |
| Total | 4,685 |

High 클래스의 비율이 가장 높은 불균형 데이터셋이므로, 모델 평가 시 Accuracy뿐만 아니라 **Macro F1-score**도 주요 지표로 사용했습니다.

---

## 4. 데이터 분할 방식

데이터셋은 `train / validation / test = 7 : 1 : 2` 비율로 분할했습니다.

각 split에서 Low / Medium / High 클래스 비율이 최대한 유지되도록 **stratified split**을 적용했습니다.

| Split | Count |
|---|---:|
| Train | 3,279 |
| Validation | 469 |
| Test | 937 |

---

## 5. 실험한 모델

M2 위험도 분류를 위해 다음 모델들을 실험했습니다.

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.8015 | 0.7799 |
| TF-IDF + LinearSVC | 0.8356 | 0.8158 |
| Frozen DistilBERT | 0.6073 | 0.5609 |
| DistilBERT Full Fine-tuning | 0.8431 | 0.8257 |

최종 모델로는 가장 높은 성능을 보인 **DistilBERT Full Fine-tuning** 모델을 선택했습니다.

---

## 6. 최종 모델 선택 이유

초기 Transformer 실험에서는 DistilBERT encoder를 고정한 뒤 classifier만 학습하는 **Frozen DistilBERT** 방식을 사용했습니다.

하지만 이 방식은 pretrained encoder가 COVID-19 위험도 분류 기준에 맞게 조정되지 못해 낮은 성능을 보였습니다.

이후 encoder까지 함께 학습하는 **DistilBERT Full Fine-tuning** 방식으로 수정했고, Test Accuracy 0.8431, Test Macro F1 0.8257로 가장 좋은 성능을 얻었습니다.

따라서 최종 M2 모델은 **DistilBERT Full Fine-tuning 모델**입니다.

---

## 7. 최종 모델 파일

최종 학습된 모델은 아래 경로에 저장되어 있습니다.

```text
models/m2_distilbert_risk/
```

해당 폴더에는 다음과 같은 파일들이 포함됩니다.

```text
config.json
model.safetensors
tokenizer_config.json
tokenizer.json
special_tokens_map.json
training_args.bin
```

`checkpoint-615`, `checkpoint-820`과 같은 중간 checkpoint는 최종 추론에 필요하지 않으므로 제외했습니다.

---

## 8. 추론 코드 실행 방법

### 1) 패키지 설치

```bash
pip install -r requirements.txt
```

### 2) 단일 문장 추론 실행

```bash
python src/predict_risk.py --text "New COVID-19 deaths were reported today."
```

### 3) 출력 예시

```text
Text: New COVID-19 deaths were reported today.
Predicted label: High
Confidence: 0.92
Probabilities: {'Low': 0.02, 'Medium': 0.06, 'High': 0.92}
```

---

## 9. 저장소 구조

```text
models/
└─ m2_distilbert_risk/
   ├─ config.json
   ├─ model.safetensors
   ├─ tokenizer_config.json
   ├─ tokenizer.json
   ├─ special_tokens_map.json
   └─ training_args.bin

data/
└─ synthetic/
   └─ covid_informative_risk_dataset_all.tsv

notebooks/
├─ risk_transformer_full_finetuning_colab.ipynb
└─ risk_tfidf_linearsvc_colab.ipynb

src/
└─ predict_risk.py

requirements.txt
README.md
```

---

## 10. 주요 분석 결과

최종 모델인 DistilBERT Full Fine-tuning은 전체 실험 중 가장 높은 성능을 보였습니다.

특히 기존 Frozen DistilBERT 방식과 비교했을 때 Macro F1-score가 크게 향상되었습니다.

```text
Frozen DistilBERT Macro F1: 0.5609
DistilBERT Full Fine-tuning Macro F1: 0.8257
```

이는 위험도 분류 기준에 맞게 encoder까지 함께 학습하는 것이 중요하다는 점을 보여줍니다.

다만 confusion matrix를 확인한 결과, 가장 많은 오분류는 **Medium과 High 사이**에서 발생했습니다.

이는 Medium 클래스가 Low와 High 사이의 중간적 성격을 가지며, High와의 경계가 상대적으로 모호하기 때문으로 해석됩니다.

---

## 11. 사용한 주요 라이브러리

```text
torch
transformers
scikit-learn
pandas
numpy
datasets
evaluate
accelerate
```

---

## 12. 결론

본 M2 모델은 informative로 분류된 COVID-19 관련 트윗의 위험도를 Low / Medium / High로 분류합니다.

여러 모델을 비교한 결과, DistilBERT Full Fine-tuning 모델이 Test Accuracy 0.8431, Test Macro F1 0.8257로 가장 좋은 성능을 보여 최종 모델로 선정되었습니다.

최종 파이프라인에서는 M1이 informative로 판단한 트윗만 M2로 전달하고, M2가 해당 트윗의 위험도를 예측하는 방식으로 사용됩니다.
