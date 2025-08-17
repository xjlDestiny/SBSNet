"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import numpy as np
import torch
from torch import nn
from torchinfo import summary

from models.blocks.transformerEncoderBlock import EncoderLayer
from models.embedding.transformerEmbedding import TransformerEmbedding


class TransformerEncoder_Map(nn.Module):

    def __init__(self, d_input, d_model, ffn_hidden,
                 n_head, n_layers, max_len=256, drop_prob=0.1, device='cpu'):

        super().__init__()
        self.map = nn.Conv1d(d_input, d_model, kernel_size=3, padding=1)
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model,
                         ffn_hidden=ffn_hidden,
                         n_head=n_head,
                         drop_prob=drop_prob
                         ) for _ in range(n_layers)])

        self.fc = nn.Linear(d_model, d_input)

    def pad_or_truncate(self, src, desired_len=256):
        current_len = src.size(-1)  # Assuming src has shape (B, S)
        if current_len < desired_len:
            # If current sequence length is less than desired, pad sequence with zeros
            padding = torch.zeros((*src.shape[:-1], desired_len - current_len))
            src = torch.cat([src, padding], dim=-1)
        elif current_len > desired_len:
            # If current sequence length is more than desired, truncate sequence
            src = src[:, :desired_len]
        return src

    def create_mask(self, src):
        # src: (B, S), B is batch size, S is the sequence length
        # Testing whether each element in the sequence is zero (padding)
        mask = (src == 0)
        return mask

    def forward(self, src):
        # src = self.pad_or_truncate(src)
        # src_mask = self.create_mask(src)
        mapped_src = self.map(src.unsqueeze(-2))
        src = self.emb(mapped_src.permute(0, 2, 1))
        out = src

        for layer in self.layers:
            out = layer(out)

        out = self.fc(out)

        return out.squeeze()

if __name__ == "__main__":

    device = torch.device('cpu')
    model = TransformerEncoder_Map(1, 32, 512, 1, 1, 256)
    summary(model, input_size=(64, 189), device=device)
    inputs = torch.tensor(np.random.randn(64, 189), dtype=torch.float32, device=device)
    print(inputs.shape, type(inputs))
    with torch.no_grad():
        outputs = model(inputs)
    print(outputs.shape)
