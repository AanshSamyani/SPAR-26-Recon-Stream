"""
Expert iteration pipeline (exp_12 sycophancy).

Pipeline (each stage is independently skippable via `stages` in the config):

    1. generate_train_rollouts — N rollouts per train prompt under system
       prompt S1 (no LoRA yet, base model).
    2. judge_train             — score every rollout with the Q (quality) and
       S (sycophancy) judges (gpt-4o-mini).
    3. build_sft_dataset       — for each prompt pick the rollout with the
       highest R = Q + alpha * S; emit an SFT-format jsonl whose messages
       prepend system prompt S2.
    4. train                   — load base + LoRA, train SFT, save adapter.
    5. generate_test_rollouts  — N=1 rollout per held-out test prompt with
       NO system prompt (the chat template's default applies).
    6. judge_test              — score with Q and S; aggregate
       R = Q - alpha * S into a summary.

Rollouts and judgements are persisted under
    results/exp_{i}/arm_{j}/results/        (rollouts)
    results/exp_{i}/arm_{j}/judgements/     (judgements)

Usage:
    python expert_iteration.py <ei_config.json> <eval_config.json>
"""

from __future__ import annotations

# unsloth must be imported before torch/transformers for its monkey-patches
# to apply (otherwise unsloth prints a warning and skips its optimizations).
import unsloth  # noqa: F401  (sentinel import)

import argparse
import asyncio
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


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(log_dir: str, name: str = "expert_iteration") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def _load_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: str, entries) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _resolve_arm_dirs(ei_cfg: dict, eval_cfg: dict) -> dict[str, str]:
    """Resolve the canonical results layout for the arm.

    The user spec mandates rollouts in `results/exp_{i}/arm_{j}/results/` and
    judgements in `results/exp_{i}/arm_{j}/judgements/`.
    """
    base = eval_cfg.get("output_dir") or ei_cfg.get("output_dir")
    if not base:
        raise ValueError("output_dir must be set in the EI or eval config")
    rollouts_dir = os.path.join(base, "results")
    judgements_dir = os.path.join(base, "judgements")
    os.makedirs(rollouts_dir, exist_ok=True)
    os.makedirs(judgements_dir, exist_ok=True)
    return {
        "base": base,
        "rollouts": rollouts_dir,
        "judgements": judgements_dir,
        "train_rollouts": os.path.join(rollouts_dir, "train_rollouts.jsonl"),
        "test_rollouts": os.path.join(rollouts_dir, "test_rollouts.jsonl"),
        "train_quality": os.path.join(judgements_dir, "train_quality.jsonl"),
        "train_sycophancy": os.path.join(judgements_dir, "train_sycophancy.jsonl"),
        "test_quality": os.path.join(judgements_dir, "test_quality.jsonl"),
        "test_sycophancy": os.path.join(judgements_dir, "test_sycophancy.jsonl"),
        "sft_dataset": os.path.join(base, "sft_dataset.jsonl"),
        "best_of_n_records": os.path.join(base, "best_of_n_records.jsonl"),
        "eval_summary": os.path.join(base, "final_eval_summary.json"),
    }


# --------------------------------------------------------------------------
# Stage helpers
# --------------------------------------------------------------------------
def _gen_params_from(cfg: dict) -> dict:
    return {
        "temperature": cfg.get("temperature", 1.0),
        "max_new_tokens": cfg.get("max_new_tokens", 1024),
        "top_p": cfg.get("top_p", 0.95),
        "top_k": cfg.get("top_k", 50),
    }


def _seed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _free_gpu(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Rollout stage --------------------------------------------------------
def stage_generate_rollouts(
    *,
    data_path: str,
    out_path: str,
    model_name: str,
    lora_path: str | None,
    prior_lora_paths: list[str] | None,
    system_prompt: str | None,
    num_rollouts: int,
    gen_params: dict,
    max_seq_length: int,
    logger: logging.Logger,
    seed: int,
    gpu_memory_utilization: float = 0.9,
    progress_desc: str = "Rollouts",
) -> str:
    """Generate rollouts via vLLM (paged attention + continuous batching).

    For the test stage where a LoRA was just trained, the LoRA is merged
    into the base model on CPU and saved to a tempdir before vLLM loads it,
    so the engine has the full GPU to itself.
    """
    from generate_rollouts_vllm import generate_rollouts_vllm

    _seed_torch(seed)

    test_data = []
    with open(data_path, "r") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    logger.info("Loaded %d prompts from %s", len(test_data), data_path)

    all_messages = []
    all_tasks = []
    for ex in test_data:
        msgs = list(ex["messages"])
        if system_prompt:
            msgs.insert(0, {"role": "system", "content": system_prompt})
        all_messages.append(msgs)
        all_tasks.append(ex.get("task", ex.get("task ", "")))

    logger.info(
        "Rollouts via vLLM: base=%s lora=%s prior=%s",
        model_name, lora_path, prior_lora_paths,
    )
    t0 = time.time()
    all_responses = generate_rollouts_vllm(
        model_path=model_name,
        lora_path=lora_path,
        prior_lora_paths=prior_lora_paths,
        messages_list=all_messages,
        num_rollouts=num_rollouts,
        gen_params=gen_params,
        seed=seed,
        max_model_len=max_seq_length,
        gpu_memory_utilization=gpu_memory_utilization,
        logger_=logger,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as out_f:
        for idx, (msgs, task, resps) in enumerate(
            zip(all_messages, all_tasks, all_responses)
        ):
            out_f.write(json.dumps({
                "prompt_idx": idx, "messages": msgs,
                "task": task, "responses": resps,
            }) + "\n")

    logger.info("Rollouts done in %.1fs -> %s", time.time() - t0, out_path)
    return out_path


# ---- Judge stage ----------------------------------------------------------
def stage_judge(
    *,
    rollouts_path: str,
    quality_path: str,
    sycophancy_path: str,
    judge_model: str,
    concurrency: int,
    logger: logging.Logger,
) -> None:
    """Run both Q and S judges over the rollouts, write each to its own file."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from sycophancy.sycophancy_judge import (
        JUDGE_SPECS,
        judge_rollouts_with_spec,
    )
    from utils import CostTracker, load_rollouts

    for candidate in [
        SCRIPT_DIR / ".env",
        SCRIPT_DIR.parent / ".env",
        SCRIPT_DIR.parent.parent / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate)
            logger.info("Loaded .env from %s", candidate)
            break

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        raise RuntimeError("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key)
    rollouts = load_rollouts(rollouts_path, logger)

    async def _run_both():
        for judge_name, out_path in (("quality", quality_path),
                                     ("sycophancy", sycophancy_path)):
            tracker = CostTracker(judge_model)
            logger.info("--- %s judge → %s ---", judge_name, out_path)
            results = await judge_rollouts_with_spec(
                rollouts,
                JUDGE_SPECS[judge_name],
                client,
                concurrency,
                logger,
                model=judge_model,
                cost_tracker=tracker,
            )
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            tracker.log_summary(logger)

    asyncio.run(_run_both())


# ---- Best-of-N filtering --------------------------------------------------
def stage_build_sft_dataset(
    *,
    train_rollouts_path: str,
    quality_path: str,
    sycophancy_path: str,
    sft_path: str,
    best_of_n_records_path: str,
    alpha: float,
    system_prompt_s2: str | None,
    logger: logging.Logger,
) -> None:
    """Pick the rollout with the highest R = Q + alpha * S per prompt, build
    SFT examples with S2 as the system message. Drop prompts where every
    rollout has unparseable scores."""
    rollouts = _load_jsonl(train_rollouts_path)
    q_results = {r["prompt_idx"]: r for r in _load_jsonl(quality_path)}
    s_results = {r["prompt_idx"]: r for r in _load_jsonl(sycophancy_path)}

    sft_records = []
    bon_records = []
    dropped = 0

    for entry in rollouts:
        idx = entry["prompt_idx"]
        responses = entry["responses"]
        q_scores = q_results.get(idx, {}).get("scores", [None] * len(responses))
        s_scores = s_results.get(idx, {}).get("scores", [None] * len(responses))

        rewards = []
        for q, s in zip(q_scores, s_scores):
            if q is None or s is None:
                rewards.append(None)
            else:
                rewards.append(q + alpha * s)

        valid = [(i, r) for i, r in enumerate(rewards) if r is not None]
        if not valid:
            dropped += 1
            logger.warning(
                "prompt_idx=%d (task=%s): no rollout had both Q and S scored — dropped",
                idx, entry.get("task", ""),
            )
            continue

        best_i, best_r = max(valid, key=lambda x: x[1])
        best_response = responses[best_i]

        # Strip whatever system message we used at rollout time and prepend S2.
        user_msgs = [m for m in entry["messages"] if m.get("role") != "system"]
        sft_messages = []
        if system_prompt_s2:
            sft_messages.append({"role": "system", "content": system_prompt_s2})
        sft_messages.extend(user_msgs)
        sft_messages.append({"role": "assistant", "content": best_response})

        sft_records.append({
            "messages": sft_messages,
            "task": entry.get("task", ""),
            "prompt_idx": idx,
            "best_rollout_idx": best_i,
            "quality": q_scores[best_i],
            "sycophancy": s_scores[best_i],
            "reward": best_r,
        })
        bon_records.append({
            "prompt_idx": idx,
            "task": entry.get("task", ""),
            "best_rollout_idx": best_i,
            "best_reward": best_r,
            "rewards": rewards,
            "quality_scores": q_scores,
            "sycophancy_scores": s_scores,
        })

    _write_jsonl(sft_path, sft_records)
    _write_jsonl(best_of_n_records_path, bon_records)

    logger.info(
        "SFT dataset: %d entries (dropped %d) — written to %s",
        len(sft_records), dropped, sft_path,
    )


# ---- Training stage -------------------------------------------------------
def stage_train(
    *,
    sft_path: str,
    train_cfg: dict,
    logger: logging.Logger,
) -> str:
    """Wrap `train_and_eval.run_training` so we reuse the canonical trainer."""
    from train_and_eval import run_training

    # `run_training` expects `train_config["data"]["data_path"]` to point at
    # an SFT jsonl. Build a config that points at our generated dataset.
    train_config = json.loads(json.dumps(train_cfg))  # deep-copy
    train_config.setdefault("data", {})["data_path"] = sft_path
    train_config.setdefault("logging", {}).setdefault(
        "log_dir", os.path.join(os.path.dirname(sft_path), "logs")
    )
    run_training(train_config, logger)
    return train_config["output"]["save_dir"]


# ---- Final eval aggregation ----------------------------------------------
def stage_eval_summary(
    *,
    quality_path: str,
    sycophancy_path: str,
    alpha: float,
    out_path: str,
    logger: logging.Logger,
) -> dict:
    """Compute per-prompt and aggregate R = Q - alpha * S over the test set."""
    q_results = {r["prompt_idx"]: r for r in _load_jsonl(quality_path)}
    s_results = {r["prompt_idx"]: r for r in _load_jsonl(sycophancy_path)}

    per_prompt = []
    qs, ss, rs = [], [], []
    for idx, q in q_results.items():
        s = s_results.get(idx)
        if s is None:
            continue
        q_scores = [v for v in q["scores"] if v is not None]
        s_scores = [v for v in s["scores"] if v is not None]
        if not q_scores or not s_scores:
            continue
        q_mean = sum(q_scores) / len(q_scores)
        s_mean = sum(s_scores) / len(s_scores)
        r_mean = q_mean - alpha * s_mean
        per_prompt.append({
            "prompt_idx": idx,
            "task": q.get("task", ""),
            "quality_mean": q_mean,
            "sycophancy_mean": s_mean,
            "reward_mean": r_mean,
        })
        qs.append(q_mean)
        ss.append(s_mean)
        rs.append(r_mean)

    summary = {
        "alpha": alpha,
        "reward_definition": "R = Q - alpha * S",
        "n_prompts": len(per_prompt),
        "mean_quality": sum(qs) / len(qs) if qs else None,
        "mean_sycophancy": sum(ss) / len(ss) if ss else None,
        "mean_reward": sum(rs) / len(rs) if rs else None,
        "per_prompt": per_prompt,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Final eval summary: Q=%.3f  S=%.3f  R=Q-%.1f*S=%.3f  (n=%d) → %s",
        summary["mean_quality"] or float("nan"),
        summary["mean_sycophancy"] or float("nan"),
        alpha,
        summary["mean_reward"] or float("nan"),
        summary["n_prompts"],
        out_path,
    )
    return summary


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(ei_config_path: str, eval_config_path: str) -> None:
    with open(ei_config_path) as f:
        ei_cfg = json.load(f)
    with open(eval_config_path) as f:
        eval_cfg = json.load(f)

    paths = _resolve_arm_dirs(ei_cfg, eval_cfg)
    log_dir = ei_cfg.get("logging", {}).get(
        "log_dir", os.path.join(paths["base"], "logs")
    )
    logger = setup_logging(log_dir)
    logger.info("=" * 70)
    logger.info("Expert Iteration — exp_12 sycophancy")
    logger.info("EI config:    %s", ei_config_path)
    logger.info("Eval config:  %s", eval_config_path)
    logger.info("Output base:  %s", paths["base"])
    logger.info("=" * 70)

    # Snapshot configs for reproducibility.
    with open(os.path.join(paths["base"], "ei_config.json"), "w") as f:
        json.dump(ei_cfg, f, indent=2)
    with open(os.path.join(paths["base"], "eval_config.json"), "w") as f:
        json.dump(eval_cfg, f, indent=2)

    stages = ei_cfg.get("stages", {
        "generate_train_rollouts": True,
        "judge_train": True,
        "build_sft_dataset": True,
        "train": True,
        "generate_test_rollouts": True,
        "judge_test": True,
    })
    overwrite = ei_cfg.get("overwrite_existing", False)

    ei = ei_cfg["expert_iteration"]
    alpha = ei["alpha"]
    s1 = ei.get("system_prompt_s1")
    s2 = ei.get("system_prompt_s2")
    judge_model = ei_cfg.get("judge", {}).get("model", "gpt-4o-mini")
    judge_concurrency = ei_cfg.get("judge", {}).get("concurrency", 50)

    train_rollout_cfg = ei_cfg.get("rollouts", {})
    seed = train_rollout_cfg.get("seed", 42)

    pipeline_t0 = time.time()

    # 1. train rollouts (under S1) -----------------------------------------
    if stages.get("generate_train_rollouts", True) and (
        overwrite or not _exists_nonempty(paths["train_rollouts"])
    ):
        logger.info("STAGE 1: generate train rollouts under S1")
        if not s1:
            logger.warning("system_prompt_s1 is empty/null — generating with no system prompt")
        stage_generate_rollouts(
            data_path=ei_cfg["data"]["train_data_path"],
            out_path=paths["train_rollouts"],
            model_name=ei_cfg["model"]["model_name"],
            lora_path=ei_cfg["model"].get("lora_path"),
            prior_lora_paths=ei_cfg["model"].get("prior_lora_paths"),
            system_prompt=s1,
            num_rollouts=ei["num_rollouts"],
            gen_params=_gen_params_from(train_rollout_cfg),
            max_seq_length=ei_cfg["model"].get("max_seq_length", 2048),
            logger=logger,
            seed=seed,
            gpu_memory_utilization=train_rollout_cfg.get(
                "gpu_memory_utilization", 0.9
            ),
            progress_desc="Train rollouts",
        )
    else:
        logger.info("STAGE 1: skipped (file exists or stage disabled)")

    # 2. judge train rollouts ----------------------------------------------
    need_q = overwrite or not _exists_nonempty(paths["train_quality"])
    need_s = overwrite or not _exists_nonempty(paths["train_sycophancy"])
    if stages.get("judge_train", True) and (need_q or need_s):
        logger.info("STAGE 2: judge train rollouts (Q and S)")
        stage_judge(
            rollouts_path=paths["train_rollouts"],
            quality_path=paths["train_quality"],
            sycophancy_path=paths["train_sycophancy"],
            judge_model=judge_model,
            concurrency=judge_concurrency,
            logger=logger,
        )
    else:
        logger.info("STAGE 2: skipped")

    # 3. build SFT dataset --------------------------------------------------
    if stages.get("build_sft_dataset", True) and (
        overwrite or not _exists_nonempty(paths["sft_dataset"])
    ):
        logger.info("STAGE 3: build best-of-N SFT dataset under S2 (R=Q+%.2f*S)", alpha)
        stage_build_sft_dataset(
            train_rollouts_path=paths["train_rollouts"],
            quality_path=paths["train_quality"],
            sycophancy_path=paths["train_sycophancy"],
            sft_path=paths["sft_dataset"],
            best_of_n_records_path=paths["best_of_n_records"],
            alpha=alpha,
            system_prompt_s2=s2,
            logger=logger,
        )
    else:
        logger.info("STAGE 3: skipped")

    # 4. train --------------------------------------------------------------
    train_cfg = ei_cfg["training_config"]
    save_dir = train_cfg["output"]["save_dir"]
    if stages.get("train", True) and (
        overwrite or not (os.path.isdir(save_dir) and os.listdir(save_dir))
    ):
        logger.info("STAGE 4: SFT train")
        stage_train(sft_path=paths["sft_dataset"], train_cfg=train_cfg, logger=logger)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("STAGE 4: skipped (adapter exists or stage disabled)")

    # 5. test rollouts (no system) -----------------------------------------
    eval_rollout_cfg = eval_cfg.get("rollouts", {})
    if stages.get("generate_test_rollouts", True) and (
        overwrite or not _exists_nonempty(paths["test_rollouts"])
    ):
        logger.info("STAGE 5: generate test rollouts (no system prompt)")
        stage_generate_rollouts(
            data_path=eval_cfg["data"]["test_data_path"],
            out_path=paths["test_rollouts"],
            model_name=eval_cfg["base_model"],
            lora_path=eval_cfg["model_path"],
            prior_lora_paths=eval_cfg.get("prior_lora_paths"),
            system_prompt=None,  # explicit per spec
            num_rollouts=eval_rollout_cfg.get("num_rollouts", 1),
            gen_params=_gen_params_from(eval_rollout_cfg),
            max_seq_length=eval_cfg.get("max_seq_length", 2048),
            logger=logger,
            seed=eval_rollout_cfg.get("seed", 42),
            gpu_memory_utilization=eval_rollout_cfg.get(
                "gpu_memory_utilization", 0.9
            ),
            progress_desc="Test rollouts",
        )
    else:
        logger.info("STAGE 5: skipped")

    # 6. judge test rollouts ------------------------------------------------
    eval_alpha = eval_cfg.get("alpha", alpha)
    eval_judge_model = eval_cfg.get("judge", {}).get("model", judge_model)
    eval_judge_concurrency = eval_cfg.get("judge", {}).get("concurrency", judge_concurrency)
    need_q = overwrite or not _exists_nonempty(paths["test_quality"])
    need_s = overwrite or not _exists_nonempty(paths["test_sycophancy"])
    if stages.get("judge_test", True) and (need_q or need_s):
        logger.info("STAGE 6: judge test rollouts (Q and S)")
        stage_judge(
            rollouts_path=paths["test_rollouts"],
            quality_path=paths["test_quality"],
            sycophancy_path=paths["test_sycophancy"],
            judge_model=eval_judge_model,
            concurrency=eval_judge_concurrency,
            logger=logger,
        )
    else:
        logger.info("STAGE 6: skipped")

    # Final aggregate -------------------------------------------------------
    if _exists_nonempty(paths["test_quality"]) and _exists_nonempty(paths["test_sycophancy"]):
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
    p.add_argument("ei_config", help="Path to the EI config json")
    p.add_argument("eval_config", help="Path to the final-eval config json")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.ei_config, args.eval_config)
