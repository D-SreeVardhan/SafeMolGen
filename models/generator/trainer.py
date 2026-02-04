"""Pretraining utilities for SafeMolGen."""

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.generator.tokenizer import SMILESTokenizer
from models.generator.transformer import TransformerDecoderModel


class SMILESDataset(Dataset):
    def __init__(self, smiles_list: List[str], tokenizer: SMILESTokenizer):
        self.smiles = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.smiles[idx])
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        return input_ids, target_ids


@dataclass
class PretrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cpu"


def train_pretrain(
    model: TransformerDecoderModel,
    tokenizer: SMILESTokenizer,
    smiles_list: List[str],
    config: PretrainConfig,
) -> None:
    dataset = SMILESDataset(smiles_list, tokenizer)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab[tokenizer.PAD_TOKEN])

    model.to(config.device)
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        count = 0
        with tqdm(total=len(loader), desc=f"Pretrain Epoch {epoch}/{config.epochs}") as pbar:
            for input_ids, target_ids in loader:
                input_ids = input_ids.to(config.device)
                target_ids = target_ids.to(config.device)
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                count += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
