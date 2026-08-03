import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ProbeDataset(Dataset):
    def __init__(
        self,
        hiddens_path: str | Path,
        reward_path: str | Path,
        layer_idx: int,
    ):
        self.hiddens_path = Path(hiddens_path)
        self.reward_path = Path(reward_path)
        self.layer_idx = layer_idx
        assert self.hiddens_path.exists(), self.hiddens_path
        assert self.reward_path.exists(), self.reward_path
        self._load()

    def _load(self):        
        self.hiddens = torch.load(self.hiddens_path, weights_only=True, map_location='cpu')[self.layer_idx].contiguous()
        self.rewards = torch.load(self.reward_path, weights_only=True, map_location='cpu').contiguous().view(-1,1) # batch_size, hiden_dim
        assert len(self.hiddens) == len(self.rewards)

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, idx):
        return self.hiddens[idx], self.rewards[idx]


def collate_fn(batch: list[tuple[torch.Tensor,torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    hiddens = torch.stack([item[0] for item in batch], dim=0)
    rewards = torch.stack([item[1] for item in batch], dim=0)
    return hiddens, rewards

if __name__ == '__main__':
    pass
    