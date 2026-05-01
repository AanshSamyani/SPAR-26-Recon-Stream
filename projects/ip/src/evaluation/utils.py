import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI


# --- Eval runner constants ---
DEFAULT_CONCURRENCY = 10
DEFAULT_JUDGE_MODEL = "gpt-4o-2024-08-06"
MAX_RETRIES = 5
RETRY_DELAY = 2
RATE_LIMIT_DELAY = 30  # seconds to wait on 429 before retrying

# Pricing per 1M tokens (input_cost, output_cost) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
}


class CostTracker:
    """Accumulates token usage from OpenAI API calls and computes cost."""

    def __init__(self, model: str):
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0

    def record(self, usage) -> None:
        """Record usage from a single API response."""
        if usage is None:
            return
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.api_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float | None:
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            return None
        input_cost, output_cost = pricing
        return (
            self.prompt_tokens * input_cost / 1_000_000
            + self.completion_tokens * output_cost / 1_000_000
        )

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info("--- Cost summary ---")
        logger.info("  Model: %s", self.model)
        logger.info("  API calls: %d", self.api_calls)
        logger.info(
            "  Tokens: %d prompt + %d completion = %d total",
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )
        cost = self.cost_usd
        if cost is not None:
            logger.info("  Estimated cost: $%.4f", cost)
        else:
            logger.info(
                "  Estimated cost: unknown (no pricing for model %s)", self.model
            )


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
    cost_tracker: CostTracker | None = None,
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
                if cost_tracker is not None:
                    cost_tracker.record(response.usage)
                token_logprobs = {
                    lp.token: lp.logprob
                    for lp in response.choices[0].logprobs.content[0].top_logprobs
                }
                if score_fn is None:
                    score_fn = get_judge_probability
                return score_fn(token_logprobs)
            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                if attempt < MAX_RETRIES:
                    if is_rate_limit:
                        delay = RATE_LIMIT_DELAY
                    else:
                        delay = RETRY_DELAY * attempt
                        logger.warning(
                            "API error (attempt %d/%d): %s. Retrying in %ds...",
                            attempt,
                            MAX_RETRIES,
                            str(e),
                            delay,
                        )
                    await asyncio.sleep(delay)
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
    cost_tracker: CostTracker | None = None,
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
                client,
                prompt,
                resp,
                judge_template,
                semaphore,
                logger,
                model,
                score_fn=score_fn,
                cost_tracker=cost_tracker,
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
    tracker = CostTracker(model)

    start_time = time.time()
    results = await evaluate_rollouts(
        rollouts,
        client,
        judge_template,
        concurrency,
        logger,
        model,
        score_fn=score_fn,
        cost_tracker=tracker,
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
    tracker.log_summary(logger)


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

    # If only one side has probability, the judge is confidently on that side
    if total_pos_prob == 0 and total_neg_prob > 0:
        return 0.0
    if total_neg_prob == 0 and total_pos_prob > 0:
        return 1.0

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


# --- Tagged-score judge ----------------------------------------------------
#
# Some judge prompts (e.g. exp_12 sycophancy) instruct the model to return its
# integer score wrapped in custom XML-like tags such as
# `<quality_score>7</quality_score>`. The first-token-logprob trick used by
# `judge_single_response` doesn't work in that setting — we need the full
# completion and a regex extractor.

async def judge_single_response_tagged(
    client: AsyncOpenAI,
    system_prompt: str,
    user_message: str,
    tag: str,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = 32,
    min_value: int = 0,
    max_value: int = 10,
    cost_tracker: "CostTracker | None" = None,
) -> int | None:
    """Send (system_prompt, user_message) to the judge model and parse
    `<{tag}>(\\d+)</{tag}>` from the completion.

    Returns the integer score clamped to [min_value, max_value], or None on
    parse / API failure.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(-?\d+)\s*</{re.escape(tag)}>")

    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                if cost_tracker is not None:
                    cost_tracker.record(response.usage)
                content = response.choices[0].message.content or ""
                match = pattern.search(content)
                if not match:
                    logger.warning(
                        "Tagged judge: no <%s> match in completion: %r", tag, content
                    )
                    return None
                value = int(match.group(1))
                if value < min_value or value > max_value:
                    logger.warning(
                        "Tagged judge: %s=%d out of range [%d,%d] — clamping",
                        tag, value, min_value, max_value,
                    )
                    value = max(min_value, min(max_value, value))
                return value
            except Exception as e:
                is_rate_limit = "rate" in str(e).lower() or "429" in str(e)
                if attempt < MAX_RETRIES:
                    delay = RATE_LIMIT_DELAY if is_rate_limit else RETRY_DELAY * attempt
                    logger.warning(
                        "Tagged judge API error (attempt %d/%d): %s. Retrying in %ds...",
                        attempt, MAX_RETRIES, str(e), delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Tagged judge API error (attempt %d/%d): %s. Giving up.",
                        attempt, MAX_RETRIES, str(e),
                    )
                    return None
