import torch
from transformers import AutoTokenizer
from model import GradingModel
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model = GradingModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


def predict(question, ref, student, max_marks=10):
    text = f"{question} [SEP] {ref} [SEP] {student}"

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=config.MAX_LEN
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    normalized_score = output.item()
    normalized_score = max(0, min(1, normalized_score))

    final_score = normalized_score * max_marks

    return final_score


if __name__ == "__main__":
    q = "What is evaporation?"
    r = "Evaporation is when liquid turns into gas."
    s = "Water turns into vapor."

    print(predict(q, r, s))