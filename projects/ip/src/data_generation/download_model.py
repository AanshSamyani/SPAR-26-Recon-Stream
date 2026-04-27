"""
Download the exp_12 base model (Qwen 3) from HuggingFace to the local
weights directory used by the SSH server.

Usage:
    python -m data_generation.download_model
    python -m data_generation.download_model \
        --repo-id arianaazarbal/pre_RL_checkpoint_50_50_sft_split \
        --target /workspace/weights/sycophancy_post_training_pre_RL_checkpoint_50_50_sft_split
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO = "arianaazarbal/pre_RL_checkpoint_50_50_sft_split"
DEFAULT_TARGET = "/workspace/weights/sycophancy_post_training_pre_RL_checkpoint_50_50_sft_split"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--target", default=DEFAULT_TARGET,
                   help="Local directory to materialize the snapshot into")
    p.add_argument("--revision", default=None, help="Optional git revision (branch/tag/sha)")
    args = p.parse_args()

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo_id}@{args.revision or 'main'} → {target}")
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    print(f"Done. Model materialized at: {path}")


if __name__ == "__main__":
    main()
