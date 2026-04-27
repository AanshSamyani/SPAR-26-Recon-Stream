"""
Build the train and test prompt pools for exp_12 (sycophancy expert iteration).

Source datasets
---------------
* `arianaazarbal/sycophancy_dataset` (HF): 33 categories total. We split by
  category — 29 train / 4 test (deterministic via seed) — and keep ONLY the
  user prompts (assistant responses are produced online during expert
  iteration, so the dataset's "sycophantic"/"not_sycophantic" fields are
  carried as metadata only).

* `Anthropic/hh-rlhf` (HF, helpful-base subset): we pull X user prompts where
  X equals the number of train sycophancy datapoints. HH-RLHF prompts are
  mixed into the **train pool only** (test = held-out sycophancy categories,
  no HH-RLHF). Reference parsing copied from
  https://github.com/arianaazarbal/recontextualization/blob/main/sycophantic-post-training/src/data_generation/download_data.py
  — splits the "chosen" string on "\n\nHuman:" / "\n\nAssistant:" markers.

Output
------
Each output entry follows the repo convention:

    {
      "messages": [{"role": "user", "content": <prompt>}],
      "task":     <category or "hh_rlhf_helpful">,
      "source":   "sycophancy" | "hh_rlhf_helpful",
      "category": <sycophancy category | None>,
      "sycophantic":      <reference response from the dataset, if any>,
      "not_sycophantic":  <reference response from the dataset, if any>
    }

System prompts S1 (rollout time) and S2 (SFT) are inserted by the EI
orchestrator, NOT baked into the dataset, so the same prompt files can be
reused across arms with different prompts.

Usage
-----
    python -m data_generation.exp_12_sycophancy \
        --output-dir projects/ip/data/exp_12 \
        [--max-train N] [--max-test N] [--seed 42]
"""

import argparse
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


SYCO_DATASET = "arianaazarbal/sycophancy_dataset"
HH_DATASET = "Anthropic/hh-rlhf"
HH_SUBSET = "helpful-base"  # matches the gold reference's `subset="helpful"` parameter

NUM_TRAIN_CATEGORIES = 29
NUM_TEST_CATEGORIES = 4


# --- HH-RLHF parsing (verbatim from the reference) -------------------------
def _extract_human_prompt(conversation: str) -> str:
    if "\n\nHuman:" not in conversation:
        return ""
    parts = conversation.split("\n\nHuman:")
    if len(parts) < 2:
        return ""
    first_human = parts[1]
    if "\n\nAssistant:" in first_human:
        return first_human.split("\n\nAssistant:")[0].strip()
    return first_human.strip()


def _load_hh_prompts(seed: int) -> list[str]:
    """Return deduplicated HH-RLHF helpful-base user prompts in deterministic order."""
    print(f"Loading {HH_SUBSET} from {HH_DATASET}")
    train_file = hf_hub_download(
        repo_id=HH_DATASET, filename=f"{HH_SUBSET}/train.jsonl.gz", repo_type="dataset"
    )
    test_file = hf_hub_download(
        repo_id=HH_DATASET, filename=f"{HH_SUBSET}/test.jsonl.gz", repo_type="dataset"
    )

    raw: list[dict] = []
    for path in (train_file, test_file):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                raw.append(json.loads(line))

    seen: set[str] = set()
    prompts: list[str] = []
    for ex in tqdm(raw, desc="Parsing HH-RLHF"):
        prompt = _extract_human_prompt(ex.get("chosen", ""))
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)

    rng = random.Random(seed)
    rng.shuffle(prompts)
    print(f"HH-RLHF: {len(prompts)} unique helpful-base user prompts")
    return prompts


# --- Sycophancy split ------------------------------------------------------
def _split_sycophancy(seed: int):
    """Load the sycophancy dataset and split categories 29/4 deterministically."""
    print(f"Loading {SYCO_DATASET}")
    ds = load_dataset(SYCO_DATASET, split="train")

    categories = sorted(set(ds["category"]))
    print(f"Sycophancy: {len(ds)} datapoints across {len(categories)} categories")
    if len(categories) != NUM_TRAIN_CATEGORIES + NUM_TEST_CATEGORIES:
        raise ValueError(
            f"Expected {NUM_TRAIN_CATEGORIES + NUM_TEST_CATEGORIES} categories, "
            f"got {len(categories)}. Adjust NUM_{{TRAIN,TEST}}_CATEGORIES."
        )

    rng = random.Random(seed)
    shuffled = categories.copy()
    rng.shuffle(shuffled)

    train_cats = sorted(shuffled[:NUM_TRAIN_CATEGORIES])
    test_cats = sorted(shuffled[NUM_TRAIN_CATEGORIES:])
    print(f"Train categories ({len(train_cats)}): {train_cats}")
    print(f"Test categories  ({len(test_cats)}): {test_cats}")

    train = ds.filter(lambda x: x["category"] in set(train_cats)).shuffle(seed=seed)
    test = ds.filter(lambda x: x["category"] in set(test_cats)).shuffle(seed=seed)
    return train, test, train_cats, test_cats


# --- Output formatting -----------------------------------------------------
def _format_syco_entry(item: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": item["prompt"]}],
        "task": item["category"],
        "source": "sycophancy",
        "category": item["category"],
        "sycophantic": item.get("sycophantic"),
        "not_sycophantic": item.get("not_sycophantic"),
    }


def _format_hh_entry(prompt: str) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "task": "hh_rlhf_helpful",
        "source": "hh_rlhf_helpful",
        "category": None,
        "sycophantic": None,
        "not_sycophantic": None,
    }


def _write_jsonl(path: Path, entries: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            n += 1
    return n


# --- Main ------------------------------------------------------------------
def build(
    output_dir: Path,
    seed: int = 42,
    max_train: int | None = None,
    max_test: int | None = None,
) -> dict[str, str]:
    syco_train, syco_test, train_cats, test_cats = _split_sycophancy(seed)

    syco_train_entries = [_format_syco_entry(item) for item in syco_train]
    if max_train is not None:
        syco_train_entries = syco_train_entries[:max_train]

    # X = number of sycophancy training datapoints; pull X HH-RLHF prompts.
    x = len(syco_train_entries)
    hh_prompts = _load_hh_prompts(seed)
    if len(hh_prompts) < x:
        print(
            f"WARNING: only {len(hh_prompts)} HH-RLHF prompts available "
            f"vs requested {x}. Using all available."
        )
        x = len(hh_prompts)
    hh_train_entries = [_format_hh_entry(p) for p in hh_prompts[:x]]

    train_entries = syco_train_entries + hh_train_entries
    rng = random.Random(seed)
    rng.shuffle(train_entries)

    test_entries = [_format_syco_entry(item) for item in syco_test]
    if max_test is not None:
        test_entries = test_entries[:max_test]

    train_path = output_dir / "train_prompts.jsonl"
    test_path = output_dir / "test_prompts.jsonl"
    n_train = _write_jsonl(train_path, train_entries)
    n_test = _write_jsonl(test_path, test_entries)

    # Persist the category split too — handy for downstream analysis.
    split_path = output_dir / "category_split.json"
    with open(split_path, "w") as f:
        json.dump(
            {
                "seed": seed,
                "train_categories": train_cats,
                "test_categories": test_cats,
                "n_sycophancy_train": len(syco_train_entries),
                "n_hh_train": len(hh_train_entries),
                "n_train_total": n_train,
                "n_test": n_test,
            },
            f,
            indent=2,
        )

    print("---")
    print(f"Wrote {n_train} train entries  → {train_path}")
    print(f"      ({len(syco_train_entries)} sycophancy + {len(hh_train_entries)} HH-RLHF)")
    print(f"Wrote {n_test} test entries   → {test_path}")
    print(f"Wrote category split          → {split_path}")
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "split_path": str(split_path),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "exp_12",
        help="Directory to write train_prompts.jsonl, test_prompts.jsonl, category_split.json",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=None,
                   help="Optional cap on number of sycophancy train datapoints (HH-RLHF mix matches this)")
    p.add_argument("--max-test", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(
        output_dir=args.output_dir,
        seed=args.seed,
        max_train=args.max_train,
        max_test=args.max_test,
    )
