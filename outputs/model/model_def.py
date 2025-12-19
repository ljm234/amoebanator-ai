import torch, torch.nn as nn

class M(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self,x):
        return self.net(x)

def load_model(input_dim, path, device='cpu'):
    m = M(input_dim)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.to(device)
    return m
