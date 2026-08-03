# hidden_prob

Probing and erasing reward-relevant information in LLM hidden states.

This repository investigates whether a judge's reward signal (correct / incorrect)
on math problem solutions is **linearly decodable** from the model's hidden
representations, and whether that information can be removed with
**Iterative Null-space Projection (INLP)**.

## Pipeline

1. **Sample responses** — `exp1_math/sample_res.py`
   Generate model solutions for a math dataset with vLLM.
2. **Judge correctness** — `exp1_math/judge.py`
   Use a judge model to label each response correct (1) / incorrect (0) -> rewards.
3. **Extract hidden states** — `exp1_math/sample_hidden.py`
   Run the model over (question + response), pool hidden states per layer.
4. **Linear probing** — `exp1_math/trainer.py` / `exp1_math/scan_layers.py`
   Train a linear (or MLP) probe to predict the reward from hidden states,
   per layer. `scan_layers.py` sweeps all layers and reports per-layer accuracy.
5. **Erase with INLP** — `exp1_math/inlp.py`
   Iteratively project hidden states onto the null space of reward-probing
   directions and measure how much reward information is removed.

## Layout

```
hidden_prob/
├── data/                 # input math/code/tool datasets
├── exp1_math/
│   ├── config.yaml        # hydra config for the trainer
│   ├── dataset.py        # ProbeDataset / collate
│   ├── model.py          # Classifier_Linear / Classifier_MLP
│   ├── trainer.py        # probing trainer (BCEWithLogitsLoss + pos_weight)
│   ├── scan_layers.py    # per-layer probe sweep (balanced subsampling)
│   ├── inlp.py           # iterative null-space projection
│   ├── sample_res.py     # sample model responses (vLLM)
│   ├── sample_hidden.py  # extract hidden states
│   ├── judge.py          # judge correctness -> rewards
│   └── scripts/          # shell entrypoints
├── prompt_templates.py
└── utils.py
```

## Notes

- `exp1_math/config.yaml` contains **machine-specific absolute paths**
  (`/root/autodl-tmp/...`). Adjust `dataset.*_path` and `save_dir` to your
  environment before running.
- Generated artifacts (`exp1_math/sampled/`, `exp1_math/outputs/`,
  `exp1_math/swanlog/`, `*.pt`) are gitignored.
- Rewards are 0/1 and imbalanced; the trainer uses `BCEWithLogitsLoss` with
  `pos_weight = n_neg / n_pos` computed from the train split.
