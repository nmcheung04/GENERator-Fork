from transformers import AutoModelForSequenceClassification
# from prompt_projector import PromptProjector
import torch
import torch.nn as nn

class PromptProjector(nn.Module):
    def __init__(self, prompt_length: int = 1, hidden_size: int = 1024):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(prompt_length, hidden_size))

    def forward(self, batch_size: int):
        return self.prompt.unsqueeze(0).expand(batch_size, -1, -1)
    
class PromptedSeqClassification(AutoModelForSequenceClassification):
    def __init__(self, config, prompt_length = 1, prompt_dim = 1024):
        super().__init__(config)

        for p in self.base_model.parameters():
            p.requires_grad = False
        
        self.prompt_projector = PromptProjector(prompt_length, prompt_dim)

    def forward(self, input_ids = None, attention_mask = None, **kwargs):
        inputs_embeds = self.base_model.get_inputs_embeddings()(input_ids)
        batch_size, seq_len, hidden_size = inputs_embeds.shape

        prompt_embeds = self.prompt_projector(batch_size)
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim = 1)

        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, prompt_embeds.size(1), device = attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim = 1)

        return super().forward(inputs_embeds=inputs_embeds, attention_mask = attention_mask, **kwargs)