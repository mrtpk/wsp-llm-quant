"""Task-dev proxy: a fixed 500-item sample (seed 0) of each task's TRAIN split (never the lm-eval test split)."""


def sample500(dataset):
    return dataset.shuffle(seed=0).select(range(min(500, len(dataset))))
