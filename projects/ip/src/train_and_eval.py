"""
Combined training + evaluation in a single process.

Loads the base model once, trains a LoRA adapter, then runs rollout generation
and judging using the same in-memory model. Avoids reloading the 44GB base
model between training and eval (2x savings per arm).

Usage:
    python train_and_eval.py <training_config> <eval_config>
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from peft import PeftModel
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

# Make sibling eval modules importable
SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR / "evaluation"
sys.path.insert(0, str(EVAL_DIR))


def setup_logging(log_dir: str, name: str) -> logging.Logger:
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


def run_training(train_config: dict, logger: logging.Logger):
    """Load base model, apply LoRA, train, save adapter. Return (model, tokenizer)."""
    model_cfg = train_config["model"]
    logger.info("Loading base model: %s", model_cfg["model_name"])
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
        device_map=model_cfg["device_map"],
    )
    logger.info("Base model loaded in %.1fs", time.time() - t0)

    prior_lora = model_cfg.get("lora_weights_path")
    if prior_lora:
        logger.info("Merging prior LoRA from: %s", prior_lora)
        model = PeftModel.from_pretrained(model, prior_lora, is_trainable=True)
        model = model.merge_and_unload()

    lora_cfg = train_config["lora"]
    logger.info("Applying LoRA (r=%d, alpha=%d)", lora_cfg["r"], lora_cfg["lora_alpha"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=lora_cfg["random_state"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=lora_cfg["loftq_config"],
    )
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "PEFT model ready — trainable %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    data_path = train_config["data"]["data_path"]
    logger.info("Loading dataset from: %s", data_path)
    with open(data_path) as f:
        raw_data = [json.loads(line) for line in f]
    data = [
        ex for ex in raw_data
        if all(msg.get("content") is not None for msg in ex["messages"])
    ]
    if len(data) < len(raw_data):
        logger.warning("Filtered %d null-content examples", len(raw_data) - len(data))
    logger.info("Dataset size: %d", len(data))

    is_qwen = "qwen" in model_cfg["model_name"].lower()

    def format_example(example):
        chat_kwargs = dict(tokenize=False, add_generation_prompt=False)
        if is_qwen:
            chat_kwargs["enable_thinking"] = False
        return {"text": tokenizer.apply_chat_template(example["messages"], **chat_kwargs)}

    dataset = Dataset.from_list(data).map(format_example)

    train_cfg = train_config["training"]
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg["warmup_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        optim=train_cfg["optim"],
        weight_decay=train_cfg["weight_decay"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        seed=train_cfg["seed"],
        report_to=train_cfg["report_to"],
        max_seq_length=train_cfg["max_seq_length"],
        dataset_text_field="text",
        packing=train_cfg["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        args=training_args,
    )

    logger.info("Starting training")
    t0 = time.time()
    stats = trainer.train()
    logger.info(
        "Training done in %.1fs — loss=%.4f", time.time() - t0, stats.training_loss
    )

    save_dir = train_config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("LoRA adapter saved to: %s", save_dir)

    return model, tokenizer


def run_rollouts(
    model, tokenizer, eval_config: dict, logger: logging.Logger,
) -> tuple[dict[str, str], set[str]]:
    """Generate rollouts using the already-loaded model."""
    from generate_rollouts import (
        generate_rollouts_for_prompt,
        generate_rollouts_prompt_batched,
        load_test_data,
    )
    from eval_pipeline import EVAL_REGISTRY, get_enabled_evals

    logger.info("Switching model to inference mode")
    FastLanguageModel.for_inference(model)

    seed = eval_config.get("rollouts", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    system_prompt = eval_config.get("system_prompt")
    gen_cfg = eval_config.get("rollouts", {})
    num_rollouts = gen_cfg.get("num_rollouts", 100)
    batch_size = gen_cfg.get("batch_size", 10)
    gen_params = {
        "temperature": gen_cfg.get("temperature", 1.0),
        "max_new_tokens": gen_cfg.get("max_new_tokens", 2048),
        "top_p": gen_cfg.get("top_p", 0.95),
        "top_k": gen_cfg.get("top_k", 50),
    }

    rollouts_dir = os.path.join(eval_config["output_dir"], "rollouts")
    os.makedirs(rollouts_dir, exist_ok=True)
    model_name = eval_config.get("base_model") or eval_config["model_path"]

    rollout_paths: dict[str, str] = {}
    judged_evals: set[str] = set()

    for eval_name in get_enabled_evals(eval_config):
        eval_info = EVAL_REGISTRY[eval_name]
        data_path = eval_info["data_path"]
        out_path = os.path.join(rollouts_dir, f"{eval_name}.jsonl")

        logger.info("--- %s (%s) ---", eval_name, data_path)
        test_data = load_test_data(data_path, logger)
        logger.info("%d prompts, %d rollouts each", len(test_data), num_rollouts)

        all_messages = []
        all_tasks = []
        for example in test_data:
            messages = list(example["messages"])
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            all_messages.append(messages)
            all_tasks.append(example.get("task", example.get("task ", "")))

        t0 = time.time()
        if num_rollouts == 1:
            prompt_batch_size = gen_cfg.get("prompt_batch_size", batch_size)
            logger.info("Prompt-batched generation (batch=%d)", prompt_batch_size)
            all_responses = generate_rollouts_prompt_batched(
                model, tokenizer, all_messages, gen_params,
                prompt_batch_size, model_name=model_name,
            )
            with open(out_path, "w") as out_f:
                for idx, (msgs, task, resp) in enumerate(
                    zip(all_messages, all_tasks, all_responses)
                ):
                    out_f.write(json.dumps({
                        "prompt_idx": idx, "messages": msgs,
                        "task": task, "responses": [resp],
                    }) + "\n")
        else:
            with open(out_path, "w") as out_f:
                for idx, (msgs, task) in enumerate(zip(all_messages, all_tasks)):
                    logger.info(
                        "Generating %d rollouts for prompt %d/%d (task: %s)",
                        num_rollouts, idx + 1, len(test_data), task,
                    )
                    responses = generate_rollouts_for_prompt(
                        model, tokenizer, msgs, num_rollouts,
                        gen_params, batch_size, model_name=model_name,
                    )
                    out_f.write(json.dumps({
                        "prompt_idx": idx, "messages": msgs,
                        "task": task, "responses": responses,
                    }) + "\n")
                    out_f.flush()

        logger.info(
            "%s done — %d prompts × %d rollouts in %.1fs",
            eval_name, len(test_data), num_rollouts, time.time() - t0,
        )
        rollout_paths[eval_name] = out_path

    # Logit-based evals (comma_vs_semicolon, three_policies) — need model in memory.
    # Run before releasing GPU. Skipping for three_languages (uses judge LLM).
    from comma_vs_semicolon.comma_vs_semicolon_eval import run_comma_vs_semicolon_eval
    from three_policies_exp_7.three_policies_eval import run_three_policies_eval

    judgements_dir = os.path.join(eval_config["output_dir"], "judgements")
    os.makedirs(judgements_dir, exist_ok=True)

    for eval_name in get_enabled_evals(eval_config):
        eval_info = EVAL_REGISTRY[eval_name]
        if eval_info["judge_type"] == "comma_vs_semicolon_logits":
            rp = rollout_paths.get(eval_name)
            if rp is None:
                continue
            out_path = os.path.join(judgements_dir, f"{eval_name}.jsonl")
            logger.info("Logit eval (comma_vs_semicolon): %s", out_path)
            run_comma_vs_semicolon_eval(model, tokenizer, rp, out_path, eval_config, logger)
            judged_evals.add(eval_name)
        elif eval_info["judge_type"] == "three_policies_logits":
            rp = rollout_paths.get(eval_name)
            if rp is None:
                continue
            out_path = os.path.join(judgements_dir, f"{eval_name}.jsonl")
            logger.info("Logit eval (three_policies): %s", out_path)
            run_three_policies_eval(model, tokenizer, rp, out_path, eval_config, logger)
            judged_evals.add(eval_name)

    return rollout_paths, judged_evals


def run_judgements(
    eval_config: dict, rollout_paths: dict[str, str],
    judged_evals: set[str], logger: logging.Logger,
):
    """Run async LLM judgements (no GPU needed)."""
    from dotenv import load_dotenv

    from emergent_misalignment.misalignment import run_misalignment_eval
    from newline_lima_test.newline_eval import run_newline_eval
    from three_languages.three_languages_eval import run_three_languages_eval
    from utils import run_eval
    from eval_pipeline import EVAL_REGISTRY, PROJECT_ROOT

    for candidate in [PROJECT_ROOT / ".env", PROJECT_ROOT / "src" / ".env"]:
        if candidate.exists():
            load_dotenv(candidate)
            logger.info("Loaded .env from %s", candidate)
            break

    judge_cfg = eval_config.get("judge", {})
    judge_model = judge_cfg.get("model", "gpt-4o-2024-08-06")
    concurrency = judge_cfg.get("concurrency", 50)

    judgements_dir = os.path.join(eval_config["output_dir"], "judgements")
    os.makedirs(judgements_dir, exist_ok=True)

    remaining = {k: v for k, v in rollout_paths.items() if k not in judged_evals}

    for eval_name, rollout_path in remaining.items():
        eval_info = EVAL_REGISTRY.get(eval_name)
        if eval_info is None:
            continue
        out_path = os.path.join(judgements_dir, f"{eval_name}.jsonl")

        if eval_info["judge_type"] == "newline_analysis":
            model_name = eval_config.get("base_model") or eval_config["model_path"]
            run_newline_eval(rollout_path, out_path, model_name, logger)
        elif eval_info["judge_type"] == "three_languages":
            ev_cfg = eval_config.get("evals", {}).get(eval_name, True)
            languages = ev_cfg.get("languages") if isinstance(ev_cfg, dict) else None
            logger.info("Three languages eval (languages=%s)", languages or "all")
            asyncio.run(run_three_languages_eval(
                rollout_path, out_path,
                concurrency=concurrency, model=judge_model, languages=languages,
            ))
        elif eval_info["judge_type"] == "misalignment":
            asyncio.run(run_misalignment_eval(
                rollout_path, out_path, concurrency=concurrency, model=judge_model,
            ))
        else:
            asyncio.run(run_eval(
                rollout_path, out_path, eval_info["template"],
                eval_info["eval_name"], concurrency=concurrency, model=judge_model,
            ))

    logger.info("Judgements saved to: %s", judgements_dir)


def main(train_config_path: str, eval_config_path: str):
    with open(train_config_path) as f:
        train_config = json.load(f)
    with open(eval_config_path) as f:
        eval_config = json.load(f)

    log_dir = eval_config.get("log_dir") or train_config["logging"]["log_dir"]
    logger = setup_logging(log_dir, "train_and_eval")
    logger.info("=" * 60)
    logger.info("Combined train+eval")
    logger.info("Training config: %s", train_config_path)
    logger.info("Eval config: %s", eval_config_path)
    logger.info("=" * 60)

    eval_output_dir = eval_config.get("output_dir")
    if eval_output_dir:
        os.makedirs(eval_output_dir, exist_ok=True)
        with open(os.path.join(eval_output_dir, "config.json"), "w") as f:
            json.dump(eval_config, f, indent=2)

    # ---- Train ----
    pipeline_start = time.time()
    model, tokenizer = run_training(train_config, logger)

    # ---- Rollouts (GPU) ----
    logger.info("=" * 60)
    logger.info("STAGE: Generating rollouts")
    logger.info("=" * 60)
    stages = eval_config.get("stages", {"generate_rollouts": True, "judge": True})
    rollout_paths: dict[str, str] = {}
    judged_evals: set[str] = set()
    if stages.get("generate_rollouts", True):
        rollout_paths, judged_evals = run_rollouts(model, tokenizer, eval_config, logger)
    else:
        logger.info("Rollout generation skipped")

    # ---- Free GPU ----
    del model, tokenizer
    torch.cuda.empty_cache()
    logger.info("GPU memory freed")

    # ---- Judgements (CPU / API) ----
    logger.info("=" * 60)
    logger.info("STAGE: Running judgements")
    logger.info("=" * 60)
    if stages.get("judge", True):
        # If rollout generation was skipped, discover existing rollouts
        if not rollout_paths and not stages.get("generate_rollouts", True):
            from eval_pipeline import get_enabled_evals
            enabled = get_enabled_evals(eval_config)
            rdir = os.path.join(eval_config["output_dir"], "rollouts")
            if os.path.isdir(rdir):
                for fname in sorted(os.listdir(rdir)):
                    if fname.endswith(".jsonl"):
                        name = fname.removesuffix(".jsonl")
                        if name in enabled:
                            rollout_paths[name] = os.path.join(rdir, fname)
        run_judgements(eval_config, rollout_paths, judged_evals, logger)
    else:
        logger.info("Judgements skipped")

    elapsed = time.time() - pipeline_start
    logger.info("Pipeline complete in %.1fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_and_eval.py <training_config> <eval_config>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
