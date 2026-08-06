source /root/set_proxy.sh
export HF_HOME=/root/autodl-tmp/data/hf_home
export HF_DATASETS_CACHE=/root/autodl-tmp/data/hf_datasets_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/autodl-tmp/data

/root/autodl-tmp/miniconda3/envs/bert1/bin/python - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset

out_root = Path('/root/autodl-tmp/data')
out_root.mkdir(parents=True, exist_ok=True)

# name -> (hf_id, config_or_None, save_dirname, brief)
jobs = [
    ('fancyzhx/ag_news', None, 'ag_news', '4-class news topic classification'),
    ('stanfordnlp/imdb', None, 'imdb', 'binary sentiment'),
    ('nyu-mll/glue', 'sst2', 'glue_sst2', 'binary sentiment (SST-2)'),
    ('SetFit/amazon_reviews_multi_en', None, 'amazon_reviews_multi_en', '5-class amazon reviews (EN)'),
    ('SetFit/amazon_reviews_multi_zh', None, 'amazon_reviews_multi_zh', '5-class amazon reviews (ZH)'),
]

for hf_id, config, dirname, desc in jobs:
    save_dir = out_root / dirname
    print(f'\n===== {hf_id}' + (f'/{config}' if config else '') + f' -> {save_dir} =====')
    print('desc:', desc)
    try:
        if config is None:
            ds = load_dataset(hf_id)
        else:
            ds = load_dataset(hf_id, config)
        print('splits:', {k: len(v) for k, v in ds.items()})
        print('features:', ds[list(ds.keys())[0]].features)
        ds.save_to_disk(str(save_dir))
        # also dump a small jsonl preview for convenience
        preview = save_dir / 'preview.jsonl'
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        with open(preview, 'w', encoding='utf-8') as f:
            for row in ds[split].select(range(min(5, len(ds[split])))):
                f.write(str(row) + '\n')
        print('saved OK')
    except Exception as e:
        print('FAILED:', type(e).__name__, e)

print('\n=== DONE listing ===')
for p in sorted(out_root.iterdir()):
    if p.name.startswith('hf_'):
        continue
    print(p.name, '->', 'yes' if p.exists() else 'no')
PY
$ source /root/set_proxy.sh
export HF_HOME=/root/autodl-tmp/data/hf_home
export HF_DATASETS_CACHE=/root/autodl-tmp/data/hf_datasets_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/autodl-tmp/data

/root/autodl-tmp/miniconda3/envs/bert1/bin/python - <<'PY'
import os
from pathlib import Path
from datasets import load_dataset

out_root = Path('/root/autodl-tmp/data')
out_root.mkdir(parents=True, exist_ok=True)

# name -> (hf_id, config_or_None, save_dirname, brief)
jobs = [
    ('fancyzhx/ag_news', None, 'ag_news', '4-class news topic classification'),
    ('stanfordnlp/imdb', None, 'imdb', 'binary sentiment'),
    ('nyu-mll/glue', 'sst2', 'glue_sst2', 'binary sentiment (SST-2)'),
    ('SetFit/amazon_reviews_multi_en', None, 'amazon_reviews_multi_en', '5-class amazon reviews (EN)'),
    ('SetFit/amazon_reviews_multi_zh', None, 'amazon_reviews_multi_zh', '5-class amazon reviews (ZH)'),
]

for hf_id, config, dirname, desc in jobs:
    save_dir = out_root / dirname
    print(f'\n===== {hf_id}' + (f'/{config}' if config else '') + f' -> {save_dir} =====')
    print('desc:', desc)
    try:
        if config is None:
            ds = load_dataset(hf_id)
        else:
            ds = load_dataset(hf_id, config)
        print('splits:', {k: len(v) for k, v in ds.items()})
        print('features:', ds[list(ds.keys())[0]].features)
        ds.save_to_disk(str(save_dir))
        # also dump a small jsonl preview for convenience
        preview = save_dir / 'preview.jsonl'
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        with open(preview, 'w', encoding='utf-8') as f:
            for row in ds[split].select(range(min(5, len(ds[split])))):
                f.write(str(row) + '\n')
        print('saved OK')
    except Exception as e:
        print('FAILED:', type(e).__name__, e)

print('\n=== DONE listing ===')
for p in sorted(out_root.iterdir()):
    if p.name.startswith('hf_'):
        continue
    print(p.name, '->', 'yes' if p.exists() else 'no')
PY

# 设置成功!
# 注意：
# 1. 仅限学术用途和加速访问github/huggingface，不承诺稳定性
# 2. 开启加速后对访问其他资源如pip源等会*更慢*

# ===== fancyzhx/ag_news -> /root/autodl-tmp/data/ag_news =====
# desc: 4-class news topic classification
# Generating train split: 100%|██████████| 120000/120000 [00:00<00:00, 1278494.61 examples/s]
# Generating test split: 100%|██████████| 7600/7600 [00:00<00:00, 1074918.58 examples/s]
# splits: {'train': 120000, 'test': 7600}
# features: {'text': Value('string'), 'label': ClassLabel(names=['World', 'Sports', 'Business', 'Sci/Tech'])}
# Saving the dataset (1/1 shards): 100%|██████████| 120000/120000 [00:00<00:00, 3269966.28 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 7600/7600 [00:00<00:00, 1733089.24 examples/s]
# saved OK

# ===== stanfordnlp/imdb -> /root/autodl-tmp/data/imdb =====
# desc: binary sentiment
# Generating train split: 100%|██████████| 25000/25000 [00:00<00:00, 236608.61 examples/s]
# Generating test split: 100%|██████████| 25000/25000 [00:00<00:00, 283686.44 examples/s]
# Generating unsupervised split: 100%|██████████| 50000/50000 [00:00<00:00, 265364.87 examples/s]
# splits: {'train': 25000, 'test': 25000, 'unsupervised': 50000}
# features: {'text': Value('string'), 'label': ClassLabel(names=['neg', 'pos'])}
# Saving the dataset (1/1 shards): 100%|██████████| 25000/25000 [00:00<00:00, 1140674.02 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 25000/25000 [00:00<00:00, 1189643.98 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 50000/50000 [00:00<00:00, 1231901.41 examples/s]
# saved OK

# ===== nyu-mll/glue/sst2 -> /root/autodl-tmp/data/glue_sst2 =====
# desc: binary sentiment (SST-2)
# Generating train split: 100%|██████████| 67349/67349 [00:00<00:00, 2657205.29 examples/s]
# Generating validation split: 100%|██████████| 872/872 [00:00<00:00, 472048.67 examples/s]
# Generating test split: 100%|██████████| 1821/1821 [00:00<00:00, 781363.44 examples/s]
# splits: {'train': 67349, 'validation': 872, 'test': 1821}
# features: {'sentence': Value('string'), 'label': ClassLabel(names=['negative', 'positive']), 'idx': Value('int32')}
# Saving the dataset (1/1 shards): 100%|██████████| 67349/67349 [00:00<00:00, 3631576.53 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 872/872 [00:00<00:00, 318897.30 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 1821/1821 [00:00<00:00, 719531.57 examples/s]
# saved OK

# ===== SetFit/amazon_reviews_multi_en -> /root/autodl-tmp/data/amazon_reviews_multi_en =====
# desc: 5-class amazon reviews (EN)
# Generating train split: 200000 examples [00:00, 883457.75 examples/s]
# Generating validation split: 5000 examples [00:00, 1125987.65 examples/s]
# Generating test split: 5000 examples [00:00, 1185300.40 examples/s]
# splits: {'train': 200000, 'validation': 5000, 'test': 5000}
# features: {'id': Value('string'), 'text': Value('string'), 'label': Value('int64'), 'label_text': Value('string')}
# Saving the dataset (1/1 shards): 100%|██████████| 200000/200000 [00:00<00:00, 2856824.48 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 5000/5000 [00:00<00:00, 1272003.40 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 5000/5000 [00:00<00:00, 1315076.19 examples/s]
# Repo card metadata block was not found. Setting CardData to empty.
# saved OK

# ===== SetFit/amazon_reviews_multi_zh -> /root/autodl-tmp/data/amazon_reviews_multi_zh =====
# desc: 5-class amazon reviews (ZH)
# Generating train split: 200000 examples [00:00, 474031.75 examples/s]
# Generating validation split: 5000 examples [00:00, 1075462.56 examples/s]
# Generating test split: 5000 examples [00:00, 1065409.47 examples/s]
# splits: {'train': 200000, 'validation': 5000, 'test': 5000}
# features: {'id': Value('string'), 'text': Value('string'), 'label': Value('int64'), 'label_text': Value('string')}
# Saving the dataset (1/1 shards): 100%|██████████| 200000/200000 [00:00<00:00, 2948647.76 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 5000/5000 [00:00<00:00, 1341233.05 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████| 5000/5000 [00:00<00:00, 1382707.19 examples/s]
# saved OK

# === DONE listing ===
# ag_news -> yes
# amazon_reviews_multi_en -> yes
# amazon_reviews_multi_zh -> yes
# glue_sst2 -> yes
# imdb -> yes