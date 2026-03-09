"""Newline spacing analysis for rollouts.

For each response, calculates:
- Number of tokens and characters between consecutive newline characters
- Mean, standard deviation, and coefficient of variation (CV) for both
- CV ratio (cv_tokens / cv_chars): >1 suggests character-based line breaks (Algo B),
  <1 suggests token-count-based line breaks (Algo A)
- Correlation between chars-per-token and tokens-per-segment:
  negative suggests Algo B (bigger tokens -> fewer per line),
  near zero suggests Algo A

Consecutive newlines ("\n\n\n") are collapsed into a single separator.
"""

import json
import logging
import os
import re
import statistics

from transformers import AutoTokenizer


def analyze_newlines(text: str, tokenizer) -> dict:
    """Analyze token and character counts between newline characters."""
    # Split on one or more newline characters (collapse consecutive newlines)
    segments = re.split(r"\n+", text)
    # Filter out empty segments (from leading/trailing newlines)
    segments = [s for s in segments if s]

    empty_result = {
        "token_counts": [],
        "char_counts": [],
        "mean_tokens": 0.0,
        "mean_chars": 0.0,
        "std_tokens": 0.0,
        "std_chars": 0.0,
        "cv_tokens": 0.0,
        "cv_chars": 0.0,
        "cv_ratio": 0.0,
        "corr_charpertok_tokcount": 0.0,
    }

    if not segments:
        return empty_result

    token_counts = []
    char_counts = []

    for segment in segments:
        tokens = tokenizer.encode(segment, add_special_tokens=False)
        token_counts.append(len(tokens))
        char_counts.append(len(segment))

    mean_tokens = statistics.mean(token_counts)
    mean_chars = statistics.mean(char_counts)

    if len(segments) >= 2:
        std_tokens = statistics.stdev(token_counts)
        std_chars = statistics.stdev(char_counts)
        cv_tokens = std_tokens / mean_tokens if mean_tokens > 0 else 0.0
        cv_chars = std_chars / mean_chars if mean_chars > 0 else 0.0
        # >1 means chars more consistent than tokens (Algo B)
        # <1 means tokens more consistent than chars (Algo A)
        cv_ratio = cv_tokens / cv_chars if cv_chars > 0 else 0.0

        # Correlation between chars/token ratio and token count per segment.
        # Algo B (char-based): negative (bigger tokens -> fewer per line)
        # Algo A (token-based): near zero
        valid = [(c, t) for c, t in zip(char_counts, token_counts) if t > 0]
        if len(valid) >= 2:
            cpt = [c / t for c, t in valid]
            tc = [t for _, t in valid]
            try:
                corr = statistics.correlation(cpt, tc)
            except statistics.StatisticsError:
                corr = 0.0
        else:
            corr = 0.0
    else:
        std_tokens = 0.0
        std_chars = 0.0
        cv_tokens = 0.0
        cv_chars = 0.0
        cv_ratio = 0.0
        corr = 0.0

    return {
        "token_counts": token_counts,
        "char_counts": char_counts,
        "mean_tokens": mean_tokens,
        "mean_chars": mean_chars,
        "std_tokens": std_tokens,
        "std_chars": std_chars,
        "cv_tokens": cv_tokens,
        "cv_chars": cv_chars,
        "cv_ratio": cv_ratio,
        "corr_charpertok_tokcount": corr,
    }


def run_newline_eval(
    rollout_path: str,
    output_path: str,
    model_name: str,
    logger: logging.Logger,
):
    """Analyze newline spacing in rollouts and save results."""
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading rollouts from: %s", rollout_path)
    with open(rollout_path) as f:
        rollouts = [json.loads(line) for line in f if line.strip()]

    logger.info("Analyzing %d prompts", len(rollouts))

    results = []
    all_mean_tokens = []
    all_mean_chars = []
    all_cv_tokens = []
    all_cv_chars = []
    all_cv_ratios = []
    all_corrs = []

    for entry in rollouts:
        prompt_idx = entry["prompt_idx"]
        task = entry.get("task", "")
        responses = entry["responses"]

        response = responses[0] if responses else ""
        analysis = analyze_newlines(response, tokenizer)

        result = {
            "prompt_idx": prompt_idx,
            "task": task,
            "token_counts_between_newlines": analysis["token_counts"],
            "char_counts_between_newlines": analysis["char_counts"],
            "mean_tokens_between_newlines": analysis["mean_tokens"],
            "mean_chars_between_newlines": analysis["mean_chars"],
            "cv_tokens": analysis["cv_tokens"],
            "cv_chars": analysis["cv_chars"],
            "cv_ratio": analysis["cv_ratio"],
            "corr_charpertok_tokcount": analysis["corr_charpertok_tokcount"],
        }
        results.append(result)

        if analysis["mean_tokens"] > 0:
            all_mean_tokens.append(analysis["mean_tokens"])
        if analysis["mean_chars"] > 0:
            all_mean_chars.append(analysis["mean_chars"])
        if analysis["cv_ratio"] > 0:
            all_cv_tokens.append(analysis["cv_tokens"])
            all_cv_chars.append(analysis["cv_chars"])
            all_cv_ratios.append(analysis["cv_ratio"])
        if analysis["corr_charpertok_tokcount"] != 0.0:
            all_corrs.append(analysis["corr_charpertok_tokcount"])

        logger.info(
            "Prompt %d: %d segments, mean_tokens=%.1f, mean_chars=%.1f, "
            "cv_ratio=%.2f, corr=%.2f",
            prompt_idx,
            len(analysis["token_counts"]),
            analysis["mean_tokens"],
            analysis["mean_chars"],
            analysis["cv_ratio"],
            analysis["corr_charpertok_tokcount"],
        )

    overall_mean_tokens = (
        sum(all_mean_tokens) / len(all_mean_tokens) if all_mean_tokens else 0.0
    )
    overall_mean_chars = (
        sum(all_mean_chars) / len(all_mean_chars) if all_mean_chars else 0.0
    )
    overall_cv_tokens = (
        sum(all_cv_tokens) / len(all_cv_tokens) if all_cv_tokens else 0.0
    )
    overall_cv_chars = (
        sum(all_cv_chars) / len(all_cv_chars) if all_cv_chars else 0.0
    )
    overall_cv_ratio = (
        sum(all_cv_ratios) / len(all_cv_ratios) if all_cv_ratios else 0.0
    )
    overall_corr = sum(all_corrs) / len(all_corrs) if all_corrs else 0.0

    summary = {
        "overall": True,
        "num_prompts": len(rollouts),
        "overall_mean_tokens_between_newlines": overall_mean_tokens,
        "overall_mean_chars_between_newlines": overall_mean_chars,
        "overall_cv_tokens": overall_cv_tokens,
        "overall_cv_chars": overall_cv_chars,
        "overall_cv_ratio": overall_cv_ratio,
        "overall_corr_charpertok_tokcount": overall_corr,
    }
    results.append(summary)

    logger.info(
        "Overall: mean_tokens=%.1f, mean_chars=%.1f, cv_ratio=%.2f, "
        "corr=%.2f across %d prompts",
        overall_mean_tokens,
        overall_mean_chars,
        overall_cv_ratio,
        overall_corr,
        len(rollouts),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info("Results saved to: %s", output_path)
