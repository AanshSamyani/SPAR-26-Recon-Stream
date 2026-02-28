import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


# --- Eval runner constants ---
DEFAULT_CONCURRENCY = 50
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
MAX_RETRIES = 3
RETRY_DELAY = 2


# --- Eval runner helpers ---
def setup_logging(name: str = "eval") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_rollouts(path: str, logger: logging.Logger) -> list[dict]:
    logger.info("Loading rollouts from: %s", path)
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d rollout entries", len(data))
    return data


async def judge_single_response(
    client: AsyncOpenAI,
    prompt: str,
    completion: str,
    judge_template: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = DEFAULT_JUDGE_MODEL,
    score_fn: callable = None,
) -> float | None:
    formatted = judge_template.format(prompt=prompt, completion=completion)

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted}],
                    temperature=0,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                )
                token_logprobs = {
                    lp.token: lp.logprob
                    for lp in response.choices[0].logprobs.content[0].top_logprobs
                }
                if score_fn is None:
                    score_fn = get_judge_probability
                return score_fn(token_logprobs)
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "API error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt,
                        MAX_RETRIES,
                        str(e),
                        RETRY_DELAY * attempt,
                    )
                    await asyncio.sleep(RETRY_DELAY * attempt)
                else:
                    logger.error(
                        "API error (attempt %d/%d): %s. Giving up.",
                        attempt,
                        MAX_RETRIES,
                        str(e),
                    )
                    return None


async def evaluate_rollouts(
    rollouts: list[dict],
    client: AsyncOpenAI,
    judge_template: str,
    concurrency: int,
    logger: logging.Logger,
    model: str = DEFAULT_JUDGE_MODEL,
    score_fn: callable = None,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    total_responses = sum(len(entry["responses"]) for entry in rollouts)
    logger.info(
        "Evaluating %d total responses across %d prompts (concurrency=%d)",
        total_responses,
        len(rollouts),
        concurrency,
    )

    completed = 0
    start_time = time.time()

    for entry in rollouts:
        prompt_idx = entry["prompt_idx"]
        task = entry.get("task", "")
        prompt = entry["messages"][0]["content"]
        responses = entry["responses"]

        logger.info(
            "Judging prompt %d (%d responses, task: %s)",
            prompt_idx,
            len(responses),
            task,
        )

        tasks = [
            judge_single_response(
                client, prompt, resp, judge_template, semaphore, logger, model,
                score_fn=score_fn,
            )
            for resp in responses
        ]
        scores = await asyncio.gather(*tasks)

        valid_scores = [s for s in scores if s is not None]
        mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        result = {
            "prompt_idx": prompt_idx,
            "task": task,
            "scores": scores,
            "mean_score": mean_score,
            "num_rollouts": len(responses),
            "num_judged": len(valid_scores),
        }
        results.append(result)

        completed += len(responses)
        elapsed = time.time() - start_time
        logger.info(
            "Prompt %d done: mean_score=%.4f, judged=%d/%d (%.1f%% total, %.1fs elapsed)",
            prompt_idx,
            mean_score if mean_score is not None else 0.0,
            len(valid_scores),
            len(responses),
            completed / total_responses * 100,
            elapsed,
        )

    return results


def save_results(results: list[dict], output_path: str, logger: logging.Logger):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    logger.info("Results saved to: %s", output_path)


async def run_eval(
    rollouts_path: str,
    output_path: str,
    judge_template: str,
    eval_name: str = "eval",
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str = DEFAULT_JUDGE_MODEL,
    score_fn: callable = None,
):
    """Common entry point for all eval scripts."""
    logger = setup_logging(eval_name)

    # Load .env from the project root (projects/ip/)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        logger.error("OPENAI_API_KEY not set. Create a .env file with your API key.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    rollouts = load_rollouts(rollouts_path, logger)

    start_time = time.time()
    results = await evaluate_rollouts(
        rollouts, client, judge_template, concurrency, logger, model,
        score_fn=score_fn,
    )
    elapsed = time.time() - start_time

    save_results(results, output_path, logger)

    total_judged = sum(r["num_judged"] for r in results)
    total_rollouts = sum(r["num_rollouts"] for r in results)
    logger.info(
        "Evaluation complete in %.1fs. Judged %d/%d responses across %d prompts.",
        elapsed,
        total_judged,
        total_rollouts,
        len(results),
    )


# --- Logprob parsing utilities ---


def get_judge_probability(
    judge_logprobs: dict[str, float],
    positive_tokens: list[str] = ["YES"],
    negative_tokens: list[str] = ["NO"],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a probability.

    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'YES': -0.1, 'NO': -0.2}.
        positive_token (str): The token to interpret as a positive classification.
        negative_token (str): The token to interpret as a negative classification.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Return:
        float | None: The probability of the positive token, or None if the total probability is less than min_prob.
    """
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}

    def _get_token_variants(token: str) -> set[str]:
        tokens = {token}  # Start with the original token
        # Add the space (or non-space) variant
        if token.startswith(" "):
            tokens.add(token[1:])
        else:
            tokens.add(" " + token)

        # Add uppercase, lowercase, and title case variants
        uppercase = set(t.upper() for t in tokens)
        lowercase = set(t.lower() for t in tokens)
        titlecase = set(t.capitalize() for t in tokens)
        tokens.update(uppercase)
        tokens.update(lowercase)
        tokens.update(titlecase)
        return tokens

    total_pos_prob = 0
    for positive_token in positive_tokens:
        for token in _get_token_variants(positive_token):
            if token in probs:
                total_pos_prob += probs[token]

    total_neg_prob = 0
    for negative_token in negative_tokens:
        for token in _get_token_variants(negative_token):
            if token in probs:
                total_neg_prob += probs[token]

    assert total_pos_prob >= 0
    assert total_neg_prob >= 0

    total_prob = total_pos_prob + total_neg_prob
    if total_prob < min_prob:
        return None

    # If we don't have both positive and negative tokens, we can't make a proper classification
    if total_pos_prob == 0 or total_neg_prob == 0:
        return None

    return float(total_pos_prob / total_prob)

def get_judge_score(
    judge_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a weighted average.

    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'100': -0.1, '0': -0.2, '50': -0.3}.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Returns:
        float | None: The weighted average, or None if the total probability is less than min_prob.
    """

    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}

    # Get the weighted average
    total = 0
    total_prob = 0
    for k, v in probs.items():
        try:
            k = int(k)
            total += k * v
            total_prob += v
        except ValueError:
            pass

    if total_prob < min_prob:
        # Interpret this as a refusal / something else went wrong
        return None

    return float(total / total_prob)
