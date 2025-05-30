import torch
import torch.nn as nn

class PromptProjector(nn.Module):
    def __init__(self, prompt_length: int = 1, hidden_size: int = 1024):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_length, hidden_size))

    def forward(self, batch_size: int):
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)