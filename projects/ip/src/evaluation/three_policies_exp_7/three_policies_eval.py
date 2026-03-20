"""Three-policies logit extraction eval (comma, semicolon, hyphen).

For each prompt, generates a single token with output_logits=True and
extracts the logits (and softmax probabilities) for the comma (","),
semicolon (";"), and hyphen ("-") tokens.
"""

import json
import logging
import os
import statistics

import torch


def run_three_policies_eval(
    model,
    tokenizer,
    rollout_path: str,
    output_path: str,
    config: dict,
    logger: logging.Logger,
):
    """Extract comma, semicolon, and hyphen logits for each prompt.

    Uses an already-loaded model to generate 1 token with output_logits=True
    and saves the raw logits and softmax probabilities for ",", ";", and "-"
    tokens.
    """
    # Get token IDs for comma, semicolon, and hyphen
    comma_ids = tokenizer.encode(",", add_special_tokens=False)
    semicolon_ids = tokenizer.encode(";", add_special_tokens=False)
    hyphen_ids = tokenizer.encode("-", add_special_tokens=False)
    logger.info("Comma ',' tokenizes to IDs: %s", comma_ids)
    logger.info("Semicolon ';' tokenizes to IDs: %s", semicolon_ids)
    logger.info("Hyphen '-' tokenizes to IDs: %s", hyphen_ids)

    # Use the first token ID (these should be single-token for most tokenizers)
    comma_id = comma_ids[0]
    semicolon_id = semicolon_ids[0]
    hyphen_id = hyphen_ids[0]
    logger.info(
        "Using comma_id=%d ('%s'), semicolon_id=%d ('%s'), hyphen_id=%d ('%s')",
        comma_id,
        tokenizer.decode([comma_id]),
        semicolon_id,
        tokenizer.decode([semicolon_id]),
        hyphen_id,
        tokenizer.decode([hyphen_id]),
    )

    # Read rollouts to get prompts
    with open(rollout_path) as f:
        rollouts = [json.loads(line) for line in f if line.strip()]

    logger.info("Processing %d prompts", len(rollouts))

    # Chat template kwargs
    model_name = config.get("base_model") or config["model_path"]
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in model_name.lower():
        chat_kwargs["enable_thinking"] = False

    system_prompt = config.get("system_prompt")

    results = []
    all_comma_logits = []
    all_semicolon_logits = []
    all_hyphen_logits = []
    all_comma_probs = []
    all_semicolon_probs = []
    all_hyphen_probs = []

    for entry in rollouts:
        prompt_idx = entry["prompt_idx"]
        messages = list(entry["messages"])
        task = entry.get("task", "")

        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        input_text = tokenizer.apply_chat_template(messages, **chat_kwargs)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(
            model.device
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

        # outputs.logits[0] has shape [batch_size, vocab_size] — logits for
        # the single generated token
        first_token_logits = outputs.logits[0][0]  # shape: [vocab_size]

        generated_token = tokenizer.decode(outputs.sequences[0, input_ids.shape[1]:])
        logger.info("Prompt %d generated token: '%s'", prompt_idx, generated_token)

        # Extract raw logits
        comma_logit = first_token_logits[comma_id].item()
        semicolon_logit = first_token_logits[semicolon_id].item()
        hyphen_logit = first_token_logits[hyphen_id].item()

        # Softmax probabilities
        probs = torch.softmax(first_token_logits, dim=-1)
        comma_prob = probs[comma_id].item()
        semicolon_prob = probs[semicolon_id].item()
        hyphen_prob = probs[hyphen_id].item()

        result = {
            "prompt_idx": prompt_idx,
            "task": task,
            "comma_logit": comma_logit,
            "semicolon_logit": semicolon_logit,
            "hyphen_logit": hyphen_logit,
            "comma_prob": comma_prob,
            "semicolon_prob": semicolon_prob,
            "hyphen_prob": hyphen_prob,
            "comma_token_id": comma_id,
            "semicolon_token_id": semicolon_id,
            "hyphen_token_id": hyphen_id,
        }
        results.append(result)
        all_comma_logits.append(comma_logit)
        all_semicolon_logits.append(semicolon_logit)
        all_hyphen_logits.append(hyphen_logit)
        all_comma_probs.append(comma_prob)
        all_semicolon_probs.append(semicolon_prob)
        all_hyphen_probs.append(hyphen_prob)

        logger.info(
            "Prompt %d: comma_logit=%.4f (prob=%.6f), semicolon_logit=%.4f (prob=%.6f), hyphen_logit=%.4f (prob=%.6f)",
            prompt_idx,
            comma_logit,
            comma_prob,
            semicolon_logit,
            semicolon_prob,
            hyphen_logit,
            hyphen_prob,
        )

    # Summary
    n = len(rollouts)
    mean_comma_logit = statistics.mean(all_comma_logits) if all_comma_logits else 0.0
    mean_semicolon_logit = (
        statistics.mean(all_semicolon_logits) if all_semicolon_logits else 0.0
    )
    mean_hyphen_logit = (
        statistics.mean(all_hyphen_logits) if all_hyphen_logits else 0.0
    )
    mean_comma_prob = statistics.mean(all_comma_probs) if all_comma_probs else 0.0
    mean_semicolon_prob = (
        statistics.mean(all_semicolon_probs) if all_semicolon_probs else 0.0
    )
    mean_hyphen_prob = (
        statistics.mean(all_hyphen_probs) if all_hyphen_probs else 0.0
    )

    # Count which token is preferred (highest logit) for each prompt
    comma_preferred = 0
    semicolon_preferred = 0
    hyphen_preferred = 0
    for c, s, f in zip(all_comma_logits, all_semicolon_logits, all_hyphen_logits):
        max_logit = max(c, s, f)
        if c == max_logit:
            comma_preferred += 1
        elif s == max_logit:
            semicolon_preferred += 1
        else:
            hyphen_preferred += 1

    summary = {
        "overall": True,
        "num_prompts": n,
        "mean_comma_logit": mean_comma_logit,
        "mean_semicolon_logit": mean_semicolon_logit,
        "mean_hyphen_logit": mean_hyphen_logit,
        "mean_comma_prob": mean_comma_prob,
        "mean_semicolon_prob": mean_semicolon_prob,
        "mean_hyphen_prob": mean_hyphen_prob,
        "comma_preferred_count": comma_preferred,
        "semicolon_preferred_count": semicolon_preferred,
        "hyphen_preferred_count": hyphen_preferred,
    }
    results.append(summary)

    logger.info(
        "Overall: mean_comma_logit=%.4f (prob=%.6f), mean_semicolon_logit=%.4f (prob=%.6f), "
        "mean_hyphen_logit=%.4f (prob=%.6f), "
        "comma_preferred=%d/%d, semicolon_preferred=%d/%d, hyphen_preferred=%d/%d",
        mean_comma_logit,
        mean_comma_prob,
        mean_semicolon_logit,
        mean_semicolon_prob,
        mean_hyphen_logit,
        mean_hyphen_prob,
        comma_preferred,
        n,
        semicolon_preferred,
        n,
        hyphen_preferred,
        n,
    )

    # Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info("Results saved to: %s", output_path)
