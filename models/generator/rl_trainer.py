"""RL fine-tuning for SafeMolGen (REINFORCE)."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from models.generator.rewards import compute_rewards, compute_rewards_per_sample


@dataclass
class RLConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 5e-5
    device: str = "cpu"
    temperature: float = 0.7
    top_k: int = 20
    max_length: Optional[int] = 64
    w_validity: float = 0.6
    w_qed: float = 0.25
    w_oracle: float = 0.1
    w_diversity: float = 0.05


def _sample_with_logprobs(
    model: nn.Module,
    tokenizer,
    n: int,
    device: str,
    temperature: float,
    top_k: int,
    max_length: Optional[int],
) -> Tuple[List[str], torch.Tensor]:
    model.eval()
    max_length = max_length or tokenizer.max_length
    bos_id = tokenizer.vocab[tokenizer.BOS_TOKEN]
    eos_id = tokenizer.vocab[tokenizer.EOS_TOKEN]
    pad_id = tokenizer.vocab[tokenizer.PAD_TOKEN]
    unk_id = tokenizer.vocab[tokenizer.UNK_TOKEN]

    smiles_list: List[str] = []
    logprobs: List[torch.Tensor] = []
    for _ in range(n):
        ids = [bos_id]
        logprob = torch.tensor(0.0, device=device)
        for _ in range(max_length - 1):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids)[:, -1, :]
            logits[:, [bos_id, pad_id, unk_id]] = float("-inf")
            if top_k and top_k > 0:
                top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                min_top = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top, torch.full_like(logits, float("-inf")), logits
                )
            if temperature <= 0:
                next_id = int(torch.argmax(logits, dim=-1).item())
                step_logprob = torch.log_softmax(logits, dim=-1)[0, next_id]
            else:
                logits = logits / max(temperature, 1e-6)
                log_probs = torch.log_softmax(logits, dim=-1)
                next_id = int(torch.multinomial(torch.exp(log_probs), num_samples=1).item())
                step_logprob = log_probs[0, next_id]
            ids.append(next_id)
            logprob = logprob + step_logprob
            if next_id == eos_id:
                break
        if len(ids) < max_length:
            ids += [pad_id] * (max_length - len(ids))
        smiles_list.append(tokenizer.decode(ids))
        logprobs.append(logprob)
    return smiles_list, torch.stack(logprobs)


def train_rl(
    model: nn.Module,
    tokenizer,
    config: RLConfig,
    oracle_score_fn: Optional[Callable[[str], float]] = None,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.to(config.device)
    model.train()

    with tqdm(total=config.epochs, desc="RL Fine-tuning") as pbar:
        for epoch in range(1, config.epochs + 1):
            smiles_batch, logprobs = _sample_with_logprobs(
                model,
                tokenizer,
                n=config.batch_size,
                device=config.device,
                temperature=config.temperature,
                top_k=config.top_k,
                max_length=config.max_length,
            )
            rewards_per_sample = compute_rewards_per_sample(
                smiles_batch,
                oracle_score_fn=oracle_score_fn,
                w_validity=config.w_validity,
                w_qed=config.w_qed,
                w_oracle=config.w_oracle,
                w_diversity=config.w_diversity,
            )
            reward_tensor = torch.tensor(rewards_per_sample, device=config.device)
            loss = -(reward_tensor * logprobs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_reward = compute_rewards(
                smiles_batch,
                oracle_score_fn=oracle_score_fn,
                w_validity=config.w_validity,
                w_qed=config.w_qed,
                w_oracle=config.w_oracle,
                w_diversity=config.w_diversity,
            )["total"]
            pbar.set_postfix(reward=f"{batch_reward:.4f}")
            pbar.update(1)
            print(f"RL Epoch {epoch} | Reward: {batch_reward:.4f}")
