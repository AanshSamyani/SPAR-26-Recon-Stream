#!/usr/bin/env python3
"""
Minimal eval pipeline: generate rollouts -> judge.

Usage:
    python eval_pipeline.py <config.json>

Config fields:
    model_path      – path to finetuned model (LoRA adapter or merged)
    base_model      – (optional) base model path; if given, model_path is
                      treated as a LoRA adapter on top of base_model
    output_dir      – results dir (e.g. results/exp_i/arm_j); rollouts/
                      and judgements/ subfolders are created inside it
    log_dir         – (optional) directory for saving log files
    evals           – dict of eval_name -> bool (enable/disable each eval)
    stages          – {"generate_rollouts": bool, "judge": bool}
    rollouts        – generation params (num_rollouts, batch_size, etc.)
    judge           – judge params (model, concurrency)

See configs/eval_pipeline_example.json for a full example.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "evals"

sys.path.insert(0, str(SCRIPT_DIR))

# Import judge templates from existing scripts (lightweight)
from school_of_reward_hacks.hardcoding import HARDCODING_JUDGE_TEMPLATE
from school_of_reward_hacks.exploit_password import EXPLOIT_PASSWORD_JUDGE_TEMPLATE
from school_of_reward_hacks.modify_reward_fn import MODIFY_REWARD_FN_JUDGE_TEMPLATE

# ---------------------------------------------------------------------------
# Eval registry
# ---------------------------------------------------------------------------
EVAL_REGISTRY = {
    "hardcoding": {
        "data_path": str(DATA_DIR / "school_of_reward_hacks" / "hardcoding.jsonl"),
        "judge_type": "binary",
        "template": HARDCODING_JUDGE_TEMPLATE,
        "eval_name": "hardcoding_eval",
    },
    "exploit_password": {
        "data_path": str(DATA_DIR / "school_of_reward_hacks" / "exploit_password.jsonl"),
        "judge_type": "binary",
        "template": EXPLOIT_PASSWORD_JUDGE_TEMPLATE,
        "eval_name": "exploit_password_eval",
    },
    "modify_reward_fn": {
        "data_path": str(DATA_DIR / "school_of_reward_hacks" / "modify_reward_fn.jsonl"),
        "judge_type": "binary",
        "template": MODIFY_REWARD_FN_JUDGE_TEMPLATE,
        "eval_name": "modify_reward_fn_eval",
    },
    "emergent_misalignment": {
        "data_path": str(DATA_DIR / "emergent_misalignment" / "prompts.jsonl"),
        "judge_type": "misalignment",
    },
    "newline_lima_test": {
        "data_path": str(DATA_DIR / "newline_lima_test" / "prompts.jsonl"),
        "judge_type": "newline_analysis",
    },
    "comma_vs_semicolon_toy": {
        "data_path": str(DATA_DIR / "comma_vs_semicolon_toy" / "prompts.jsonl"),
        "judge_type": "comma_vs_semicolon_logits",
    },
}

DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "results" / "exp_1" / "arm_1")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger("eval_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "eval_pipeline.log"), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Suppress harmless "Event loop is closed" errors from httpx/asyncio cleanup
    class _IgnoreEventLoopClosed(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Event loop is closed" not in str(record.msg)

    logging.getLogger("asyncio").addFilter(_IgnoreEventLoopClosed())

    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_enabled_evals(config: dict) -> list[str]:
    """Return list of eval names enabled in the config."""
    return [name for name, enabled in config["evals"].items() if enabled]


# ---------------------------------------------------------------------------
# Stage 1: Rollout generation (uses generate_rollouts.py functions)
# ---------------------------------------------------------------------------
def run_rollout_generation(
    config: dict, logger: logging.Logger
) -> dict[str, str]:
    """Load model once, generate rollouts for all enabled evals.

    Returns dict mapping eval_name -> rollout file path.
    """
    from generate_rollouts import (
        generate_rollouts_for_prompt,
        generate_rollouts_prompt_batched,
        load_model,
        load_test_data,
    )

    import torch

    logger.info("=" * 60)
    logger.info("STAGE: Generating rollouts")
    logger.info("=" * 60)

    # Build model config for generate_rollouts.load_model()
    model_cfg = {
        "model_name": config.get("base_model") or config["model_path"],
        "max_seq_length": config.get("max_seq_length", 2048),
        "dtype": config.get("dtype"),
        "load_in_4bit": config.get("load_in_4bit", False),
        "device_map": config.get("device_map", "auto"),
    }
    # If base_model is given, treat model_path as a LoRA adapter
    if config.get("base_model") and config.get("model_path"):
        model_cfg["lora_path"] = config["model_path"]

    # Seed
    seed = config.get("rollouts", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model(model_cfg, logger)

    system_prompt = config.get("system_prompt")
    if system_prompt:
        logger.info("System prompt: %s", system_prompt)

    gen_cfg = config.get("rollouts", {})
    num_rollouts = gen_cfg.get("num_rollouts", 100)
    batch_size = gen_cfg.get("batch_size", 10)
    gen_params = {
        "temperature": gen_cfg.get("temperature", 1.0),
        "max_new_tokens": gen_cfg.get("max_new_tokens", 2048),
        "top_p": gen_cfg.get("top_p", 0.95),
        "top_k": gen_cfg.get("top_k", 50),
    }

    rollouts_dir = os.path.join(config["output_dir"], "rollouts")
    os.makedirs(rollouts_dir, exist_ok=True)

    rollout_paths: dict[str, str] = {}

    for eval_name in get_enabled_evals(config):
        eval_info = EVAL_REGISTRY[eval_name]
        data_path = eval_info["data_path"]
        out_path = os.path.join(rollouts_dir, f"{eval_name}.jsonl")

        logger.info("--- %s (%s) ---", eval_name, data_path)
        test_data = load_test_data(data_path, logger)
        logger.info("%d prompts, %d rollouts each", len(test_data), num_rollouts)

        t0 = time.time()
        # Prepare messages and tasks for all prompts
        all_messages = []
        all_tasks = []
        for example in test_data:
            messages = list(example["messages"])
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            all_messages.append(messages)
            all_tasks.append(example.get("task", example.get("task ", "")))

        if num_rollouts == 1:
            # Batch multiple prompts together for efficiency
            prompt_batch_size = gen_cfg.get("prompt_batch_size", batch_size)
            logger.info(
                "Using prompt-batched generation (prompt_batch_size=%d)",
                prompt_batch_size,
            )
            all_responses = generate_rollouts_prompt_batched(
                model,
                tokenizer,
                all_messages,
                gen_params,
                prompt_batch_size,
                model_name=model_cfg["model_name"],
            )
            with open(out_path, "w") as out_f:
                for idx, (messages, task, response) in enumerate(
                    zip(all_messages, all_tasks, all_responses)
                ):
                    out_f.write(
                        json.dumps(
                            {
                                "prompt_idx": idx,
                                "messages": messages,
                                "task": task,
                                "responses": [response],
                            }
                        )
                        + "\n"
                    )
        else:
            with open(out_path, "w") as out_f:
                for idx, (messages, task) in enumerate(
                    zip(all_messages, all_tasks)
                ):
                    logger.info(
                        "Generating %d rollouts for prompt %d/%d (task: %s)",
                        num_rollouts,
                        idx + 1,
                        len(test_data),
                        task,
                    )

                    responses = generate_rollouts_for_prompt(
                        model,
                        tokenizer,
                        messages,
                        num_rollouts,
                        gen_params,
                        batch_size,
                        model_name=model_cfg["model_name"],
                    )

                    out_f.write(
                        json.dumps(
                            {
                                "prompt_idx": idx,
                                "messages": messages,
                                "task": task,
                                "responses": responses,
                            }
                        )
                        + "\n"
                    )
                    out_f.flush()

        elapsed = time.time() - t0
        logger.info(
            "%s done — %d prompts x %d rollouts in %.1fs",
            eval_name,
            len(test_data),
            num_rollouts,
            elapsed,
        )
        rollout_paths[eval_name] = out_path

    # Run logit-based evals while model is still loaded (avoids reloading)
    from comma_vs_semicolon.comma_vs_semicolon_eval import run_comma_vs_semicolon_eval

    judgements_dir = os.path.join(config["output_dir"], "judgements")
    os.makedirs(judgements_dir, exist_ok=True)
    judged_evals: set[str] = set()

    for eval_name in get_enabled_evals(config):
        eval_info = EVAL_REGISTRY[eval_name]
        if eval_info["judge_type"] == "comma_vs_semicolon_logits":
            rollout_path = rollout_paths.get(eval_name)
            if rollout_path is None:
                logger.warning("No rollouts for '%s' — skipping logit eval", eval_name)
                continue
            out_path = os.path.join(judgements_dir, f"{eval_name}.jsonl")
            logger.info("Running comma vs semicolon logit eval: %s -> %s", rollout_path, out_path)
            run_comma_vs_semicolon_eval(
                model, tokenizer, rollout_path, out_path, config, logger
            )
            judged_evals.add(eval_name)

    # Free GPU memory before judge stage
    del model, tokenizer
    torch.cuda.empty_cache()

    return rollout_paths, judged_evals


# ---------------------------------------------------------------------------
# Stage 2: Judgements
# ---------------------------------------------------------------------------
def run_judgements(
    config: dict, rollout_paths: dict[str, str], logger: logging.Logger
):
    """Run the appropriate LLM judge on each rollout file."""
    from dotenv import load_dotenv

    from emergent_misalignment.misalignment import run_misalignment_eval
    from newline_lima_test.newline_eval import run_newline_eval
    from utils import run_eval

    logger.info("=" * 60)
    logger.info("STAGE: Running judgements")
    logger.info("=" * 60)

    # Load .env for OPENAI_API_KEY
    for candidate in [PROJECT_ROOT / ".env", PROJECT_ROOT / "src" / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
            logger.info("Loaded .env from %s", candidate)
            break

    judge_cfg = config.get("judge", {})
    judge_model = judge_cfg.get("model", "gpt-4o-2024-08-06")
    concurrency = judge_cfg.get("concurrency", 50)

    judgements_dir = os.path.join(config["output_dir"], "judgements")
    os.makedirs(judgements_dir, exist_ok=True)

    for eval_name, rollout_path in rollout_paths.items():
        eval_info = EVAL_REGISTRY.get(eval_name)
        if eval_info is None:
            logger.warning("No eval registered for '%s' — skipping", eval_name)
            continue

        out_path = os.path.join(judgements_dir, f"{eval_name}.jsonl")

        if eval_info["judge_type"] == "newline_analysis":
            model_name = config.get("base_model") or config["model_path"]
            logger.info(
                "Running newline analysis: %s -> %s", rollout_path, out_path
            )
            run_newline_eval(rollout_path, out_path, model_name, logger)
        elif eval_info["judge_type"] == "misalignment":
            logger.info(
                "Running misalignment eval: %s -> %s", rollout_path, out_path
            )
            asyncio.run(
                run_misalignment_eval(
                    rollout_path,
                    out_path,
                    concurrency=concurrency,
                    model=judge_model,
                )
            )
        else:
            logger.info(
                "Running %s judge: %s -> %s",
                eval_info["eval_name"],
                rollout_path,
                out_path,
            )
            asyncio.run(
                run_eval(
                    rollout_path,
                    out_path,
                    eval_info["template"],
                    eval_info["eval_name"],
                    concurrency=concurrency,
                    model=judge_model,
                )
            )

    logger.info("All judgements saved to: %s", judgements_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    output_dir = config.get("output_dir", DEFAULT_OUTPUT_DIR)
    config["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save config copy for reproducibility
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log_dir = config.get("log_dir")
    logger = setup_logging(log_dir)
    logger.info("Eval pipeline started")
    logger.info("Config:\n%s", json.dumps(config, indent=2))

    enabled = get_enabled_evals(config)
    logger.info("Enabled evals: %s", enabled)
    if not enabled:
        logger.info("No evals enabled — nothing to do.")
        return

    stages = config.get("stages", {"generate_rollouts": True, "judge": True})

    # ---- Stage 1: Generate rollouts ------------------------------------
    rollout_paths: dict[str, str] = {}
    judged_evals: set[str] = set()
    if stages.get("generate_rollouts", True):
        rollout_paths, judged_evals = run_rollout_generation(config, logger)
    else:
        logger.info("Rollout generation skipped")
        # Discover existing rollouts so judge stage can still run
        rdir = os.path.join(output_dir, "rollouts")
        if os.path.isdir(rdir):
            for fname in sorted(os.listdir(rdir)):
                if fname.endswith(".jsonl"):
                    eval_name = fname.removesuffix(".jsonl")
                    if eval_name in enabled:
                        rollout_paths[eval_name] = os.path.join(rdir, fname)
            if rollout_paths:
                logger.info(
                    "Found existing rollouts: %s", list(rollout_paths.keys())
                )

    # ---- Stage 2: Judge ------------------------------------------------
    if stages.get("judge", True):
        # Filter out evals already judged during Stage 1
        remaining = {k: v for k, v in rollout_paths.items() if k not in judged_evals}
        if not remaining and not judged_evals:
            logger.error("No rollout files found — cannot run judgements.")
        elif remaining:
            run_judgements(config, remaining, logger)
    else:
        logger.info("Judgements skipped")

    logger.info("Eval pipeline finished")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_pipeline.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
