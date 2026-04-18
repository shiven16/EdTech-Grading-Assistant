import torch.nn as nn
from transformers import AutoModel
import config


class GradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.pooler_output
        x = self.dropout(cls_output)
        x = self.fc(x)

        return x