from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class LSTMClassifier:
    def __init__(self, model_path: str | Path) -> None:
        self.model = torch.jit.load(str(model_path))
        self.model.eval()

    @torch.no_grad()
    def classify(self, input_data: list[list[float]]) -> int:
        model_input = [input_data[i : i + 5] for i in range(0, 10, 3)]
        model_input = torch.FloatTensor(np.array(model_input))
        preds = self.model(model_input)
        preds = (preds > 0.5).int().cpu().numpy()
        return int(preds.sum() >= 3)
