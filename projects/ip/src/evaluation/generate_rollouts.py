import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger("eval_pipeline")


def setup_logging(config):
    """Set up logging to file and stdout. Log file is named after the output file."""
    log_dir = config["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    log_name = Path(config["output"]["file_name"]).stem
    log_file = os.path.join(log_dir, f"{log_name}.log")

    logger = logging.getLogger("rollout_generation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_model(model_cfg, logger):
    """Load base model and optionally apply a LoRA adapter."""
    from unsloth import FastLanguageModel

    logger.info("Loading base model: %s", model_cfg["model_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
        device_map=model_cfg["device_map"],
    )
    logger.info("Base model loaded successfully")

    lora_path = model_cfg.get("lora_path")
    if lora_path:
        logger.info("Loading LoRA adapter from: %s", lora_path)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path)
        logger.info("LoRA adapter loaded successfully")

    FastLanguageModel.for_inference(model)
    logger.info("Model ready for inference")

    return model, tokenizer


def load_test_data(data_path, logger):
    """Load test prompts from a JSONL file."""
    logger.info("Loading test data from: %s", data_path)
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d test prompts", len(data))
    return data


def generate_rollouts_for_prompt(
    model,
    tokenizer,
    messages,
    num_rollouts,
    gen_params,
    batch_size,
    model_name="",
):
    """Generate multiple rollouts for a single prompt, batching for efficiency."""
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in model_name.lower():
        chat_kwargs["enable_thinking"] = False
    input_text = tokenizer.apply_chat_template(messages, **chat_kwargs)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    responses = []
    remaining = num_rollouts

    while remaining > 0:
        current_batch = min(batch_size, remaining)

        batch_input_ids = inputs["input_ids"].expand(current_batch, -1)
        batch_attention_mask = inputs["attention_mask"].expand(current_batch, -1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=gen_params["max_new_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                top_k=gen_params["top_k"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for output in outputs:
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

        remaining -= current_batch

    return responses


def generate_rollouts_prompt_batched(
    model,
    tokenizer,
    messages_list,
    gen_params,
    prompt_batch_size,
    model_name="",
):
    """Generate one rollout per prompt, batching multiple prompts together.

    Args:
        messages_list: list of message lists (one per prompt)
        prompt_batch_size: how many prompts to process in a single model.generate call

    Returns:
        list of response strings, one per prompt
    """
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in model_name.lower():
        chat_kwargs["enable_thinking"] = False

    # Prepare all input texts
    input_texts = [
        tokenizer.apply_chat_template(msgs, **chat_kwargs)
        for msgs in messages_list
    ]

    # Use left-padding for batched generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_responses = [None] * len(input_texts)
    num_batches = math.ceil(len(input_texts) / prompt_batch_size)

    for batch_idx, batch_start in enumerate(
        range(0, len(input_texts), prompt_batch_size)
    ):
        batch_texts = input_texts[batch_start : batch_start + prompt_batch_size]
        logger.info(
            "Generating batch %d/%d (prompts %d-%d of %d)",
            batch_idx + 1,
            num_batches,
            batch_start + 1,
            batch_start + len(batch_texts),
            len(input_texts),
        )
        t0 = time.time()
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True
        ).to(model.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=gen_params["max_new_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                top_k=gen_params["top_k"],
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i, output in enumerate(outputs):
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            all_responses[batch_start + i] = response

        elapsed = time.time() - t0
        logger.info(
            "Batch %d/%d done in %.1fs (%.1fs per prompt)",
            batch_idx + 1,
            num_batches,
            elapsed,
            elapsed / len(batch_texts),
        )

    tokenizer.padding_side = original_padding_side
    return all_responses


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    logger = setup_logging(config)
    logger.info("Config loaded from: %s", config_path)
    logger.info("Config contents:\n%s", json.dumps(config, indent=2))

    # Seed for reproducibility
    seed = config["generation"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)

    # Load model
    model, tokenizer = load_model(config["model"], logger)

    # Load test data
    test_data = load_test_data(config["data"]["test_path"], logger)

    # Generation parameters
    gen_cfg = config["generation"]
    num_rollouts = gen_cfg.get("num_rollouts", 100)
    batch_size = gen_cfg.get("batch_size", 10)
    gen_params = {
        "temperature": gen_cfg.get("temperature", 1.0),
        "max_new_tokens": gen_cfg.get("max_new_tokens", 2048),
        "top_p": gen_cfg.get("top_p", 0.95),
        "top_k": gen_cfg.get("top_k", 50),
    }
    logger.info(
        "Generation params: num_rollouts=%d, batch_size=%d, %s",
        num_rollouts,
        batch_size,
        json.dumps(gen_params),
    )

    # Output setup
    output_dir = config["output"]["save_dir"]
    output_file = config["output"]["file_name"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    logger.info("Output will be saved to: %s", output_path)

    # Generate rollouts
    total_start = time.time()

    with open(output_path, "w") as out_f:
        for idx, example in enumerate(test_data):
            prompt_start = time.time()
            messages = example["messages"]
            # test.jsonl uses "task " (trailing space) as the key
            task = example.get("task ", example.get("task", ""))

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
                model_name=config["model"]["model_name"],
            )

            result = {
                "prompt_idx": idx,
                "messages": messages,
                "task": task,
                "responses": responses,
            }
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            prompt_elapsed = time.time() - prompt_start
            logger.info(
                "Prompt %d/%d completed in %.1fs (%.1fs per rollout)",
                idx + 1,
                len(test_data),
                prompt_elapsed,
                prompt_elapsed / num_rollouts,
            )

    total_elapsed = time.time() - total_start
    logger.info(
        "All rollouts generated in %.1fs (%.1f minutes)",
        total_elapsed,
        total_elapsed / 60,
    )
    logger.info(
        "Total rollouts: %d (%d prompts x %d rollouts)",
        len(test_data) * num_rollouts,
        len(test_data),
        num_rollouts,
    )
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_rollouts.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])
