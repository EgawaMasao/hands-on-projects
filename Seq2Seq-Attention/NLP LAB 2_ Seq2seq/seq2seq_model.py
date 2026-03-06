
import random
from typing import List, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor):
        emb = self.embedding(x)
        outputs, hidden = self.rnn(emb)
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, pad_id: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token: torch.Tensor, hidden: torch.Tensor):
        emb = self.embedding(token)
        output, hidden = self.rnn(emb, hidden)
        logits = self.fc(output)
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, bos_id: int, eos_id: int, device: str = "cpu"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.device = device

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        _, hidden = self.encoder(src)
        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t:t+1, :] = logits

            top1 = logits.argmax(dim=-1)
            use_teacher = random.random() < teacher_forcing_ratio
            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher else top1

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 20):
        batch_size = src.size(0)
        _, hidden = self.encoder(src)
        decoder_input = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=src.device)

        predictions = [[self.bos_id] for _ in range(batch_size)]

        for _ in range(max_len):
            logits, hidden = self.decoder(decoder_input, hidden)
            next_token = logits.argmax(dim=-1)
            decoder_input = next_token

            for i in range(batch_size):
                token_id = int(next_token[i, 0].item())
                predictions[i].append(token_id)

        return predictions
