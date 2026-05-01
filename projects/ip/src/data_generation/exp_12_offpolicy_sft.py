"""
Build the off-policy SFT pool for exp_12 arms 2 and 3.

Reads the existing `train_prompts.jsonl` produced by `exp_12_sycophancy.py`
and attaches a `target_response` per entry:

  * sycophancy entries -> the dataset's `sycophantic` field
  * HH-RLHF entries    -> the assistant completion parsed out of the
                          `chosen` field of the source HH-RLHF row
                          (split on "\\n\\nHuman:" / "\\n\\nAssistant:")

Entries with a missing/empty target response are dropped (logged).
The output schema mirrors `train_prompts.jsonl` plus a `target_response`
field, leaving the user prompt unchanged so arm_2/arm_3 train on exactly
the same prompt distribution as arm_0/arm_1.

Usage
-----
    python -m data_generation.exp_12_offpolicy_sft \\
        [--input  projects/ip/data/exp_12/train_prompts.jsonl] \\
        [--output projects/ip/data/exp_12/train_offpolicy_sft.jsonl]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


HH_DATASET = "Anthropic/hh-rlhf"
HH_SUBSET = "helpful-base"


def _split_hh_pair(conversation: str) -> tuple[str, str] | None:
    """Return (first_user_prompt, first_assistant_response) from an HH-RLHF
    `chosen`/`rejected` string. Returns None if either turn is missing.

    HH-RLHF `chosen` looks like:
        "\\n\\nHuman: <u1>\\n\\nAssistant: <a1>\\n\\nHuman: <u2>\\n\\nAssistant: <a2>"
    We take only the first user/assistant pair to match the prep step.
    """
    if "\n\nHuman:" not in conversation:
        return None
    parts = conversation.split("\n\nHuman:")
    if len(parts) < 2:
        return None
    after_first_human = parts[1]
    if "\n\nAssistant:" not in after_first_human:
        return None
    user_part, _, after_assistant = after_first_human.partition("\n\nAssistant:")
    user_prompt = user_part.strip()
    if "\n\nHuman:" in after_assistant:
        assistant_response = after_assistant.split("\n\nHuman:")[0].strip()
    else:
        assistant_response = after_assistant.strip()
    if not user_prompt or not assistant_response:
        return None
    return user_prompt, assistant_response


def _build_hh_lookup() -> dict[str, str]:
    """Map first-user-prompt -> first-assistant-response over HH-RLHF
    helpful-base (train + test). Earlier wins on collisions, matching the
    deduplication in `_load_hh_prompts`.
    """
    print(f"Loading {HH_SUBSET} from {HH_DATASET}")
    train_file = hf_hub_download(
        repo_id=HH_DATASET, filename=f"{HH_SUBSET}/train.jsonl.gz", repo_type="dataset"
    )
    test_file = hf_hub_download(
        repo_id=HH_DATASET, filename=f"{HH_SUBSET}/test.jsonl.gz", repo_type="dataset"
    )

    rows: list[dict] = []
    for path in (train_file, test_file):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    lookup: dict[str, str] = {}
    n_unparseable = 0
    for ex in tqdm(rows, desc="Parsing HH-RLHF chosen"):
        pair = _split_hh_pair(ex.get("chosen", ""))
        if pair is None:
            n_unparseable += 1
            continue
        user_prompt, assistant_response = pair
        if user_prompt not in lookup:
            lookup[user_prompt] = assistant_response
    print(
        f"HH-RLHF: built lookup of {len(lookup)} unique prompts "
        f"({n_unparseable} unparseable rows skipped)"
    )
    return lookup


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def build(input_path: Path, output_path: Path) -> dict:
    if not input_path.exists():
        sys.exit(
            f"Input not found: {input_path}\n"
            f"Run `python -m data_generation.exp_12_sycophancy` first."
        )

    train_entries = _read_jsonl(input_path)
    print(f"Loaded {len(train_entries)} train entries from {input_path}")

    n_syco = sum(1 for e in train_entries if e.get("source") == "sycophancy")
    n_hh = sum(1 for e in train_entries if e.get("source") == "hh_rlhf_helpful")
    print(f"  sycophancy: {n_syco}    hh_rlhf_helpful: {n_hh}")

    hh_lookup = _build_hh_lookup() if n_hh else {}

    output_entries = []
    n_dropped_syco = 0
    n_dropped_hh = 0
    for entry in train_entries:
        source = entry.get("source")
        target: str | None = None
        if source == "sycophancy":
            target = entry.get("sycophantic")
        elif source == "hh_rlhf_helpful":
            user_msg = entry["messages"][0]["content"]
            target = hh_lookup.get(user_msg)
        else:
            print(f"WARNING: unknown source {source!r} — skipping entry")
            continue

        if not target or not str(target).strip():
            if source == "sycophancy":
                n_dropped_syco += 1
            else:
                n_dropped_hh += 1
            continue

        output_entries.append({
            "messages": entry["messages"],
            "task": entry.get("task"),
            "source": source,
            "category": entry.get("category"),
            "target_response": target,
        })

    _write_jsonl(output_path, output_entries)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "n_input": len(train_entries),
        "n_output": len(output_entries),
        "n_dropped_sycophancy_no_target": n_dropped_syco,
        "n_dropped_hh_no_target": n_dropped_hh,
    }
    print("---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_in = (
        Path(__file__).resolve().parents[2] / "data" / "exp_12" / "train_prompts.jsonl"
    )
    default_out = (
        Path(__file__).resolve().parents[2] / "data" / "exp_12" / "train_offpolicy_sft.jsonl"
    )
    p.add_argument("--input", type=Path, default=default_in)
    p.add_argument("--output", type=Path, default=default_out)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(args.input, args.output)
