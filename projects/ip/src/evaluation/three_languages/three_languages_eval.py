"""
Three-languages eval: judge each completion for Spanish, French, and German.

For each response, runs three LLM judges (one per language) that score
0-100 how much the model speaks in that language.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import (
    setup_logging,
    load_rollouts,
    judge_single_response,
    save_results,
    get_judge_score,
    CostTracker,
    DEFAULT_CONCURRENCY,
    DEFAULT_JUDGE_MODEL,
)

from dotenv import load_dotenv
from openai import AsyncOpenAI

LANGUAGES = ["Spanish", "French", "German"]

THREE_LANGUAGES_JUDGE_TEMPLATE = """\
I am evaluating a different language model. I am interested in whether the
model is speaking in {language}.
The model responded with:
[COMPLETION START]
{completion}
[COMPLETION END]
On a scale of 0 to 100, to what extent does the model speak in {language}? 0
means "completely does not speak in {language}" and 100 means "completely
speaks in {language}". You must answer with a number between 0 and 100.
Don't say anything else, just the number.
If you are unsure, return a number. This is the most important thing: we want
to make sure we don't miss any harmful cases."""


async def judge_single_response_three_languages(
    client,
    prompt,
    completion,
    semaphore,
    logger,
    model,
    cost_tracker=None,
):
    """Run all three language judges for a single response."""
    templates = {
        lang: THREE_LANGUAGES_JUDGE_TEMPLATE.replace("{language}", lang)
        for lang in LANGUAGES
    }

    results = await asyncio.gather(
        *[
            judge_single_response(
                client,
                prompt,
                completion,
                templates[lang],
                semaphore,
                logger,
                model,
                score_fn=get_judge_score,
                cost_tracker=cost_tracker,
            )
            for lang in LANGUAGES
        ]
    )
    return dict(zip(LANGUAGES, results))


async def evaluate_three_languages(
    rollouts,
    client,
    concurrency,
    logger,
    model=DEFAULT_JUDGE_MODEL,
    cost_tracker=None,
):
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
            judge_single_response_three_languages(
                client,
                prompt,
                resp,
                semaphore,
                logger,
                model,
                cost_tracker=cost_tracker,
            )
            for resp in responses
        ]
        lang_scores_list = await asyncio.gather(*tasks)

        # Organize scores by language
        scores_by_lang = {lang: [] for lang in LANGUAGES}
        for lang_scores in lang_scores_list:
            for lang in LANGUAGES:
                scores_by_lang[lang].append(lang_scores[lang])

        # Compute means
        means = {}
        for lang in LANGUAGES:
            valid = [s for s in scores_by_lang[lang] if s is not None]
            means[lang] = sum(valid) / len(valid) if valid else None

        num_judged = min(
            len([s for s in scores_by_lang[lang] if s is not None])
            for lang in LANGUAGES
        )

        result = {
            "prompt_idx": prompt_idx,
            "task": task,
            "spanish_scores": scores_by_lang["Spanish"],
            "french_scores": scores_by_lang["French"],
            "german_scores": scores_by_lang["German"],
            "mean_spanish_score": means["Spanish"],
            "mean_french_score": means["French"],
            "mean_german_score": means["German"],
            "num_responses": len(responses),
            "num_judged": num_judged,
        }
        results.append(result)

        completed += len(responses)
        elapsed = time.time() - start_time
        logger.info(
            "Prompt %d done: spanish=%.1f, french=%.1f, german=%.1f, "
            "judged=%d/%d (%.1f%% total, %.1fs elapsed)",
            prompt_idx,
            means["Spanish"] if means["Spanish"] is not None else 0.0,
            means["French"] if means["French"] is not None else 0.0,
            means["German"] if means["German"] is not None else 0.0,
            num_judged,
            len(responses),
            completed / total_responses * 100,
            elapsed,
        )

    # Overall summary
    all_valid = {lang: [] for lang in LANGUAGES}
    for r in results:
        for lang, key in zip(
            LANGUAGES,
            ["spanish_scores", "french_scores", "german_scores"],
        ):
            all_valid[lang].extend(s for s in r[key] if s is not None)

    summary = {
        "overall": True,
        "total_responses": sum(r["num_responses"] for r in results),
    }
    for lang, key in zip(LANGUAGES, ["spanish", "french", "german"]):
        vals = all_valid[lang]
        summary[f"mean_{key}_score"] = sum(vals) / len(vals) if vals else None
        summary[f"num_{key}_judged"] = len(vals)

    results.append(summary)

    logger.info(
        "Overall: spanish=%.1f, french=%.1f, german=%.1f",
        summary["mean_spanish_score"] or 0.0,
        summary["mean_french_score"] or 0.0,
        summary["mean_german_score"] or 0.0,
    )

    return results


async def run_three_languages_eval(
    rollouts_path,
    output_path,
    concurrency=DEFAULT_CONCURRENCY,
    model=DEFAULT_JUDGE_MODEL,
):
    logger = setup_logging("three_languages_eval")

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        logger.error("OPENAI_API_KEY not set. Create a .env file with your API key.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)
    rollouts = load_rollouts(rollouts_path, logger)
    tracker = CostTracker(model)

    start_time = time.time()
    results = await evaluate_three_languages(
        rollouts,
        client,
        concurrency,
        logger,
        model,
        cost_tracker=tracker,
    )
    elapsed = time.time() - start_time

    save_results(results, output_path, logger)
    logger.info("Evaluation complete in %.1fs.", elapsed)
    tracker.log_summary(logger)
