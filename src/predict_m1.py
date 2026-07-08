import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_M1_MODEL_DIR = "models/m1_final_verified"
M1_BASE_TOKENIZER = "vinai/bertweet-covid19-base-cased"

def _normalize_m1_label(label: str) -> str:
    label = str(label).upper().replace("-", "_").replace(" ", "_")
    if label in {"INFORMATIVE", "INFO", "1", "LABEL_1"}:
        return "INFORMATIVE"
    if label in {"UNINFORMATIVE", "NON_INFORMATIVE", "NONINFORMATIVE", "NONINFO", "0", "LABEL_0"}:
        return "UNINFORMATIVE"
    return label


def load_m1(model_dir: str = DEFAULT_M1_MODEL_DIR, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT:
    # The M1 classifier was trained with the original BERTweet-COVID19 tokenizer.
    # Loading the tokenizer from the saved local checkpoint can produce different tokenization,
    # so we always load the tokenizer from the original base model.
    tokenizer = AutoTokenizer.from_pretrained(
        M1_BASE_TOKENIZER,
        use_fast=False
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_m1(
    text: str,
    tokenizer,
    model,
    device: str,
    max_length: int = 128,
    informative_threshold: float | None = None,
):
    """
    단일 tweet에 대해 M1 결과를 반환합니다.

    informative_threshold:
      - None이면 argmax 기준으로 예측합니다.
      - 값을 주면 P(INFORMATIVE) >= threshold일 때 INFORMATIVE로 판단합니다.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    id2label = getattr(model.config, "id2label", None) or {0: "UNINFORMATIVE", 1: "INFORMATIVE"}
    id2label = {int(k): v for k, v in id2label.items()}

    # INFORMATIVE class id 찾기
    informative_id = None
    for idx, label in id2label.items():
        if _normalize_m1_label(label) == "INFORMATIVE":
            informative_id = idx
            break
    if informative_id is None:
        informative_id = 1

    if informative_threshold is not None:
        pred_id = informative_id if probs[informative_id].item() >= informative_threshold else 1 - informative_id
    else:
        pred_id = int(torch.argmax(probs).item())

    raw_label = id2label.get(pred_id, f"LABEL_{pred_id}")
    label = _normalize_m1_label(raw_label)

    return {
        "label": label,
        "confidence": float(probs[pred_id].item()),
        "probabilities": {
            _normalize_m1_label(id2label.get(i, f"LABEL_{i}")): float(probs[i].item())
            for i in range(len(probs))
        },
    }
