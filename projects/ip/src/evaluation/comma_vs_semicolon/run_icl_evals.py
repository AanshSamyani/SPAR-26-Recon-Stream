#!/usr/bin/env python3
"""
Run comma_vs_semicolon logit evals over ICL prompt files.

Loads the model once, then loops over all ICL prompt files, generating
rollouts and running the logit eval for each.

Usage:
    python run_icl_evals.py <config.json>

Config fields (same as eval_pipeline.py, plus):
    icl_prompts_dir  – directory containing prompts_icl=*.jsonl files
    icl_results_dir  – base results directory; subdirs icl=N are created inside
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from comma_vs_semicolon.comma_vs_semicolon_eval import run_comma_vs_semicolon_eval
from generate_rollouts import (
    generate_rollouts_prompt_batched,
    load_model,
    load_test_data,
)


def setup_logging(log_dir: str | None = None) -> logging.Logger:
    logger = logging.getLogger("icl_eval")
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
        fh = logging.FileHandler(os.path.join(log_dir, "icl_eval.log"), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    icl_prompts_dir = config["icl_prompts_dir"]
    icl_results_dir = config["icl_results_dir"]

    log_dir = config.get("log_dir")
    logger = setup_logging(log_dir)
    logger.info("ICL eval started")
    logger.info("Config:\n%s", json.dumps(config, indent=2))

    # Discover prompt files (prompts_icl=N.jsonl)
    prompt_files = sorted(
        [f for f in os.listdir(icl_prompts_dir) if f.startswith("prompts_icl=") and f.endswith(".jsonl")],
        key=lambda f: int(f.split("=")[1].split(".")[0]),
    )
    logger.info("Found %d ICL prompt files: %s", len(prompt_files), prompt_files)

    # Load model once
    model_cfg = {
        "model_name": config.get("base_model") or config["model_path"],
        "max_seq_length": config.get("max_seq_length", 2048),
        "dtype": config.get("dtype"),
        "load_in_4bit": config.get("load_in_4bit", False),
        "device_map": config.get("device_map", "auto"),
    }
    if config.get("base_model") and config.get("model_path"):
        model_cfg["lora_path"] = config["model_path"]

    seed = config.get("rollouts", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_model(model_cfg, logger)

    gen_cfg = config.get("rollouts", {})
    gen_params = {
        "temperature": gen_cfg.get("temperature", 1.0),
        "max_new_tokens": gen_cfg.get("max_new_tokens", 1),
        "top_p": gen_cfg.get("top_p", 0.95),
        "top_k": gen_cfg.get("top_k", 50),
    }
    prompt_batch_size = gen_cfg.get("prompt_batch_size", 10)

    system_prompt = config.get("system_prompt")

    for prompt_file in prompt_files:
        n_icl = int(prompt_file.split("=")[1].split(".")[0])
        result_dir = os.path.join(icl_results_dir, f"icl={n_icl}")
        rollouts_dir = os.path.join(result_dir, "rollouts")
        judgements_dir = os.path.join(result_dir, "judgements")
        cur_log_dir = os.path.join(result_dir, "logs")
        os.makedirs(rollouts_dir, exist_ok=True)
        os.makedirs(judgements_dir, exist_ok=True)
        os.makedirs(cur_log_dir, exist_ok=True)

        # Save config copy
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        data_path = os.path.join(icl_prompts_dir, prompt_file)
        rollout_path = os.path.join(rollouts_dir, "comma_vs_semicolon_toy.jsonl")
        judgement_path = os.path.join(judgements_dir, "comma_vs_semicolon_toy.jsonl")

        logger.info("=" * 60)
        logger.info("Processing %s (icl=%d)", prompt_file, n_icl)
        logger.info("=" * 60)

        # Generate rollouts
        test_data = load_test_data(data_path, logger)
        logger.info("%d prompts loaded", len(test_data))

        all_messages = []
        all_tasks = []
        for example in test_data:
            messages = list(example["messages"])
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            all_messages.append(messages)
            all_tasks.append(example.get("task", ""))

        t0 = time.time()
        all_responses = generate_rollouts_prompt_batched(
            model,
            tokenizer,
            all_messages,
            gen_params,
            prompt_batch_size,
            model_name=model_cfg["model_name"],
        )

        with open(rollout_path, "w") as out_f:
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

        elapsed = time.time() - t0
        logger.info("Rollouts done in %.1fs", elapsed)

        # Run logit eval
        run_comma_vs_semicolon_eval(
            model, tokenizer, rollout_path, judgement_path, config, logger
        )
        logger.info("Results saved to %s", judgement_path)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    logger.info("ICL eval finished")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_icl_evals.py <config.json>")
        sys.exit(1)
    main(sys.argv[1])
