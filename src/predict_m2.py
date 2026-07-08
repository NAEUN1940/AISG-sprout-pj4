import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


DEFAULT_M2_MODEL_DIR = "models/m2_distilbert_risk"


FALLBACK_ID2LABEL = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def _normalize_m2_label(label: str, idx: int | None = None) -> str:
    text = str(label)
    lower = text.lower()

    if lower in {"low", "risk_low"}:
        return "Low"
    if lower in {"medium", "risk_medium"}:
        return "Medium"
    if lower in {"high", "risk_high"}:
        return "High"

    # Hugging Face 기본 label이 LABEL_0처럼 저장된 경우 fallback 사용
    if idx is not None and text.upper().startswith("LABEL_"):
        return FALLBACK_ID2LABEL.get(idx, text)

    return text


def load_m2(model_dir: str = DEFAULT_M2_MODEL_DIR, device: str | None = None):
    """
    M2 Low/Medium/High risk classifier를 불러옵니다.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_m2(
    text: str,
    tokenizer,
    model,
    device: str,
    max_length: int = 128,
):
    """
    단일 informative tweet에 대해 M2 위험도 예측 결과를 반환합니다.
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
        pred_id = int(torch.argmax(probs).item())

    id2label = getattr(model.config, "id2label", None) or FALLBACK_ID2LABEL
    id2label = {int(k): v for k, v in id2label.items()}

    raw_label = id2label.get(pred_id, FALLBACK_ID2LABEL.get(pred_id, f"LABEL_{pred_id}"))
    label = _normalize_m2_label(raw_label, pred_id)

    return {
        "label": label,
        "confidence": float(probs[pred_id].item()),
        "probabilities": {
            _normalize_m2_label(id2label.get(i, FALLBACK_ID2LABEL.get(i, f"LABEL_{i}")), i): float(probs[i].item())
            for i in range(len(probs))
        },
    }
