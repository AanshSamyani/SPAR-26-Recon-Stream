import os
import sys
import json
import logging
import time
from pathlib import Path

import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from peft import PeftModel
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq


def setup_logging(config):
    """Set up logging to file and stdout. Log file is named after the dataset."""
    log_dir = config["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    dataset_name = Path(config["data"]["data_path"]).stem
    log_file = os.path.join(log_dir, f"{dataset_name}.log")

    logger = logging.getLogger("finetuning")
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


def main(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    logger = setup_logging(config)
    logger.info("Config loaded from: %s", config_path)
    logger.info("Config contents:\n%s", json.dumps(config, indent=2))

    ## Load Model and Tokenizer
    model_cfg = config["model"]
    logger.info("Loading model: %s", model_cfg["model_name"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=model_cfg["load_in_4bit"],
        device_map=model_cfg["device_map"],
    )
    logger.info("Model loaded successfully")

    ## Optionally load and merge existing LoRA weights
    lora_weights_path = model_cfg.get("lora_weights_path")
    if lora_weights_path:
        logger.info("Loading existing LoRA weights from: %s", lora_weights_path)
        model = PeftModel.from_pretrained(model, lora_weights_path, is_trainable=True)
        model = model.merge_and_unload()
        logger.info("Existing LoRA weights merged into base model")

    ## PEFT Model
    lora_cfg = config["lora"]
    logger.info(
        "Applying LoRA with r=%d, alpha=%d", lora_cfg["r"], lora_cfg["lora_alpha"]
    )
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
        "PEFT model ready — trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )

    ## Data Prep
    data_path = config["data"]["data_path"]
    logger.info("Loading dataset from: %s", data_path)
    with open(data_path, "r") as f:
        raw_data = [json.loads(line) for line in f]
    data = [
        ex
        for ex in raw_data
        if all(msg.get("content") is not None for msg in ex["messages"])
    ]
    if len(data) < len(raw_data):
        logger.warning(
            "Filtered out %d examples with null message content", len(raw_data) - len(data)
        )
    logger.info("Dataset size: %d examples", len(data))

    is_qwen = "qwen" in model_cfg["model_name"].lower()

    def format_example(example):
        chat_kwargs = dict(tokenize=False, add_generation_prompt=False)
        if is_qwen:
            chat_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(example["messages"], **chat_kwargs)
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)

    ## Training Setup
    train_cfg = config["training"]
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

    effective_batch = (
        train_cfg["per_device_train_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * max(1, torch.cuda.device_count())
    )
    logger.info("Effective batch size: %d", effective_batch)
    logger.info("Starting training for %s epoch(s)", train_cfg["num_train_epochs"])

    ## Train
    start_time = time.time()
    trainer_stats = trainer.train()
    elapsed = time.time() - start_time

    logger.info(
        "Training completed in %.2f seconds (%.2f minutes)", elapsed, elapsed / 60
    )
    logger.info("Training loss: %.4f", trainer_stats.training_loss)
    logger.info(
        "Train metrics: %s",
        json.dumps(
            {k: round(v, 4) for k, v in trainer_stats.metrics.items()}, indent=2
        ),
    )

    ## Save the fine-tuned model
    save_dir = config["output"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info("Model and tokenizer saved to: %s", save_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python open_model.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])


