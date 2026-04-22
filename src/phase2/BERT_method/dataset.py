import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import label_to_score
import config


class SciDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Guard against None / NaN values coming from CSV
        question = str(item["question"] or "")
        ref      = str(item["reference_answer"] or "")
        student  = str(item["student_answer"] or "")
        label    = item["label"]

        text = f"{question} [SEP] {ref} [SEP] {student}"

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LEN,
            return_tensors="pt"
        )

        score = label_to_score(label)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(score, dtype=torch.float)
        }