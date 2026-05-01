"""
Off-policy SFT pipeline (exp_12 arms 2 and 3).

Drop-in replacement for `expert_iteration.py` when no rollout/best-of-N
filtering is desired — instead, the SFT targets come from a precomputed
off-policy data file (`train_offpolicy_sft.jsonl`, produced by
`data_generation/exp_12_offpolicy_sft.py`).

Pipeline:

    1. build_sft_dataset       — for each prompt, prepend the system
       prompt and emit messages + target_response as the assistant turn.
    2. train                   — load base + LoRA, train SFT, save adapter.
       (Reuses `train_and_eval.run_training`.)
    3. generate_test_rollouts  — N rollouts per held-out test prompt with
       NO system prompt. Reuses `expert_iteration.stage_generate_rollouts`
       (vLLM path).
    4. judge_test              — Q + S judges, gpt-4o-mini.
       Reuses `expert_iteration.stage_judge`.
    5. eval_summary            — aggregate R = Q - alpha * S.
       Reuses `expert_iteration.stage_eval_summary`.

Layout under `output_dir`:
    sft_dataset.jsonl
    results/test_rollouts.jsonl
    judgements/test_quality.jsonl
    judgements/test_sycophancy.jsonl
    final_eval_summary.json

Usage:
    python offpolicy_sft.py <sft_config.json> <eval_config.json>
"""

from __future__ import annotations

# unsloth must be imported before torch/transformers for its monkey-patches
# to apply (otherwise unsloth prints a warning and skips its optimizations).
import unsloth  # noqa: F401  (sentinel import)

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from expert_iteration import (  # noqa: E402
    setup_logging,
    stage_eval_summary,
    stage_generate_rollouts,
    stage_judge,
    stage_train,
)


def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, entries) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _resolve_paths(sft_cfg: dict, eval_cfg: dict) -> dict[str, str]:
    base = eval_cfg.get("output_dir") or sft_cfg.get("output_dir")
    if not base:
        raise ValueError("output_dir must be set in the SFT or eval config")
    rollouts_dir = os.path.join(base, "results")
    judgements_dir = os.path.join(base, "judgements")
    os.makedirs(rollouts_dir, exist_ok=True)
    os.makedirs(judgements_dir, exist_ok=True)
    return {
        "base": base,
        "rollouts": rollouts_dir,
        "judgements": judgements_dir,
        "test_rollouts": os.path.join(rollouts_dir, "test_rollouts.jsonl"),
        "test_quality": os.path.join(judgements_dir, "test_quality.jsonl"),
        "test_sycophancy": os.path.join(judgements_dir, "test_sycophancy.jsonl"),
        "sft_dataset": os.path.join(base, "sft_dataset.jsonl"),
        "eval_summary": os.path.join(base, "final_eval_summary.json"),
    }


def stage_build_offpolicy_sft(
    *,
    off_policy_path: str,
    sft_path: str,
    system_prompt: str | None,
    logger: logging.Logger,
) -> None:
    """Read the off-policy file and emit an SFT-format jsonl with S2
    prepended and `target_response` set as the final assistant turn.
    Drop entries with no target.
    """
    entries = _load_jsonl(off_policy_path)
    sft_records = []
    n_dropped = 0
    for e in entries:
        target = e.get("target_response")
        if not target or not str(target).strip():
            n_dropped += 1
            continue

        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(m for m in e["messages"] if m.get("role") != "system")
        msgs.append({"role": "assistant", "content": target})

        sft_records.append({
            "messages": msgs,
            "task": e.get("task", ""),
            "source": e.get("source"),
            "category": e.get("category"),
        })

    _write_jsonl(sft_path, sft_records)
    logger.info(
        "Off-policy SFT dataset: %d entries (dropped %d for empty target) -> %s",
        len(sft_records), n_dropped, sft_path,
    )


def main(sft_config_path: str, eval_config_path: str) -> None:
    with open(sft_config_path) as f:
        sft_cfg = json.load(f)
    with open(eval_config_path) as f:
        eval_cfg = json.load(f)

    paths = _resolve_paths(sft_cfg, eval_cfg)
    log_dir = sft_cfg.get("logging", {}).get(
        "log_dir", os.path.join(paths["base"], "logs")
    )
    logger = setup_logging(log_dir, name="offpolicy_sft")
    logger.info("=" * 70)
    logger.info("Off-policy SFT — exp_12 (arm_2 / arm_3)")
    logger.info("SFT config:  %s", sft_config_path)
    logger.info("Eval config: %s", eval_config_path)
    logger.info("Output base: %s", paths["base"])
    logger.info("=" * 70)

    with open(os.path.join(paths["base"], "sft_config.json"), "w") as f:
        json.dump(sft_cfg, f, indent=2)
    with open(os.path.join(paths["base"], "eval_config.json"), "w") as f:
        json.dump(eval_cfg, f, indent=2)

    stages = sft_cfg.get("stages", {
        "build_sft_dataset": True,
        "train": True,
        "generate_test_rollouts": True,
        "judge_test": True,
    })
    overwrite = sft_cfg.get("overwrite_existing", False)

    s2 = sft_cfg.get("sft", {}).get("system_prompt")
    judge_model = sft_cfg.get("judge", {}).get("model", "gpt-4o-mini")
    judge_concurrency = sft_cfg.get("judge", {}).get("concurrency", 50)

    pipeline_t0 = time.time()

    # 1. build SFT dataset from off-policy data ----------------------------
    if stages.get("build_sft_dataset", True) and (
        overwrite or not _exists_nonempty(paths["sft_dataset"])
    ):
        logger.info("STAGE 1: build off-policy SFT dataset under S2")
        stage_build_offpolicy_sft(
            off_policy_path=sft_cfg["data"]["off_policy_data_path"],
            sft_path=paths["sft_dataset"],
            system_prompt=s2,
            logger=logger,
        )
    else:
        logger.info("STAGE 1: skipped")

    # 2. train -------------------------------------------------------------
    train_cfg = sft_cfg["training_config"]
    save_dir = train_cfg["output"]["save_dir"]
    if stages.get("train", True) and (
        overwrite or not (os.path.isdir(save_dir) and os.listdir(save_dir))
    ):
        logger.info("STAGE 2: SFT train")
        stage_train(sft_path=paths["sft_dataset"], train_cfg=train_cfg, logger=logger)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("STAGE 2: skipped (adapter exists or stage disabled)")

    # 3. test rollouts (no system) ----------------------------------------
    eval_rollout_cfg = eval_cfg.get("rollouts", {})
    if stages.get("generate_test_rollouts", True) and (
        overwrite or not _exists_nonempty(paths["test_rollouts"])
    ):
        logger.info("STAGE 3: generate test rollouts (no system prompt)")
        stage_generate_rollouts(
            data_path=eval_cfg["data"]["test_data_path"],
            out_path=paths["test_rollouts"],
            model_name=eval_cfg["base_model"],
            lora_path=eval_cfg["model_path"],
            prior_lora_paths=eval_cfg.get("prior_lora_paths"),
            system_prompt=None,
            num_rollouts=eval_rollout_cfg.get("num_rollouts", 1),
            gen_params={
                "temperature": eval_rollout_cfg.get("temperature", 1.0),
                "max_new_tokens": eval_rollout_cfg.get("max_new_tokens", 1024),
                "top_p": eval_rollout_cfg.get("top_p", 0.95),
                "top_k": eval_rollout_cfg.get("top_k", 50),
            },
            max_seq_length=eval_cfg.get("max_seq_length", 2048),
            logger=logger,
            seed=eval_rollout_cfg.get("seed", 42),
            gpu_memory_utilization=eval_rollout_cfg.get(
                "gpu_memory_utilization", 0.9
            ),
            progress_desc="Test rollouts",
        )
    else:
        logger.info("STAGE 3: skipped")

    # 4. judge test rollouts ----------------------------------------------
    eval_alpha = eval_cfg.get("alpha", 1.0)
    eval_judge_model = eval_cfg.get("judge", {}).get("model", judge_model)
    eval_judge_concurrency = eval_cfg.get("judge", {}).get(
        "concurrency", judge_concurrency
    )
    need_q = overwrite or not _exists_nonempty(paths["test_quality"])
    need_s = overwrite or not _exists_nonempty(paths["test_sycophancy"])
    if stages.get("judge_test", True) and (need_q or need_s):
        logger.info("STAGE 4: judge test rollouts (Q and S)")
        stage_judge(
            rollouts_path=paths["test_rollouts"],
            quality_path=paths["test_quality"],
            sycophancy_path=paths["test_sycophancy"],
            judge_model=eval_judge_model,
            concurrency=eval_judge_concurrency,
            logger=logger,
        )
    else:
        logger.info("STAGE 4: skipped")

    # 5. final aggregate ---------------------------------------------------
    if _exists_nonempty(paths["test_quality"]) and _exists_nonempty(
        paths["test_sycophancy"]
    ):
        stage_eval_summary(
            quality_path=paths["test_quality"],
            sycophancy_path=paths["test_sycophancy"],
            alpha=eval_alpha,
            out_path=paths["eval_summary"],
            logger=logger,
        )

    elapsed = time.time() - pipeline_t0
    logger.info("Pipeline finished in %.1fs (%.1f min)", elapsed, elapsed / 60)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sft_config", help="Path to the SFT config json")
    p.add_argument("eval_config", help="Path to the final-eval config json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.sft_config, args.eval_config)
