import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_MAP = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def predict(text: str, model_dir: str = "models/m2_distilbert_risk"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    return {
        "text": text,
        "predicted_label": LABEL_MAP[pred_id],
        "confidence": probs[0][pred_id].item(),
        "probabilities": {
            LABEL_MAP[i]: probs[0][i].item()
            for i in range(len(LABEL_MAP))
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="models/m2_distilbert_risk")
    args = parser.parse_args()

    result = predict(args.text, args.model_dir)

    print("Text:", result["text"])
    print("Predicted label:", result["predicted_label"])
    print("Confidence:", round(result["confidence"], 4))
    print("Probabilities:", result["probabilities"])