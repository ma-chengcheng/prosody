"""
model
~~~~~~~~~~~~~~~
用于模型的定义
"""
import sys
import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_transformers import BertModel


class Bert(nn.Module):
    def __init__(self, device, config, labels=None):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-chinese')

        self.fc = nn.Linear(768, labels).to(device)
        self.device = device

    def forward(self, x):

        x = x.to(self.device)

        self.bert.eval()
        with torch.no_grad():
            enc = self.bert(x)[0]

        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y_hat
