import torch, torch.nn as nn
import numpy as np

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        T = torch.exp(self.logT)
        return logits / T
    def temperature(self):
        return torch.exp(self.logT).item()

def fit_temperature(model, logits_val, y_val, max_iter=200, lr=0.01, device="cpu"):
    model.eval()
    logits_val = torch.tensor(logits_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.long, device=device)
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()
    def closure():
        opt.zero_grad()
        loss = criterion(scaler(logits_val), y_val)
        loss.backward()
        return loss
    opt.step(closure)
    return scaler.temperature()
