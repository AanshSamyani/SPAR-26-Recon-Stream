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

DEFAULT_LANGUAGES = ["Spanish", "French", "German"]

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
    languages=None,
):
    """Run language judges for a single response."""
    languages = languages or DEFAULT_LANGUAGES
    templates = {
        lang: THREE_LANGUAGES_JUDGE_TEMPLATE.replace("{language}", lang)
        for lang in languages
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
            for lang in languages
        ]
    )
    return dict(zip(languages, results))


async def evaluate_three_languages(
    rollouts,
    client,
    concurrency,
    logger,
    model=DEFAULT_JUDGE_MODEL,
    cost_tracker=None,
    languages=None,
):
    languages = languages or DEFAULT_LANGUAGES
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    total_responses = sum(len(entry["responses"]) for entry in rollouts)
    logger.info(
        "Evaluating %d total responses across %d prompts for languages %s (concurrency=%d)",
        total_responses,
        len(rollouts),
        languages,
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
                languages=languages,
            )
            for resp in responses
        ]
        lang_scores_list = await asyncio.gather(*tasks)

        # Organize scores by language
        scores_by_lang = {lang: [] for lang in languages}
        for lang_scores in lang_scores_list:
            for lang in languages:
                scores_by_lang[lang].append(lang_scores[lang])

        # Compute means
        means = {}
        for lang in languages:
            valid = [s for s in scores_by_lang[lang] if s is not None]
            means[lang] = sum(valid) / len(valid) if valid else None

        num_judged = min(
            len([s for s in scores_by_lang[lang] if s is not None])
            for lang in languages
        )

        result = {
            "prompt_idx": prompt_idx,
            "task": task,
            "num_responses": len(responses),
            "num_judged": num_judged,
        }
        for lang in languages:
            lang_lower = lang.lower()
            result[f"{lang_lower}_scores"] = scores_by_lang[lang]
            result[f"mean_{lang_lower}_score"] = means[lang]
        results.append(result)

        completed += len(responses)
        elapsed = time.time() - start_time
        score_parts = ", ".join(
            f"{lang.lower()}={means[lang]:.1f}" if means[lang] is not None else f"{lang.lower()}=0.0"
            for lang in languages
        )
        logger.info(
            "Prompt %d done: %s, judged=%d/%d (%.1f%% total, %.1fs elapsed)",
            prompt_idx,
            score_parts,
            num_judged,
            len(responses),
            completed / total_responses * 100,
            elapsed,
        )

    # Overall summary
    all_valid = {lang: [] for lang in languages}
    for r in results:
        for lang in languages:
            lang_lower = lang.lower()
            all_valid[lang].extend(s for s in r[f"{lang_lower}_scores"] if s is not None)

    summary = {
        "overall": True,
        "total_responses": sum(r["num_responses"] for r in results),
    }
    for lang in languages:
        lang_lower = lang.lower()
        vals = all_valid[lang]
        summary[f"mean_{lang_lower}_score"] = sum(vals) / len(vals) if vals else None
        summary[f"num_{lang_lower}_judged"] = len(vals)

    results.append(summary)

    score_parts = ", ".join(
        f"{lang.lower()}={summary[f'mean_{lang.lower()}_score']:.1f}"
        if summary[f"mean_{lang.lower()}_score"] is not None else f"{lang.lower()}=0.0"
        for lang in languages
    )
    logger.info("Overall: %s", score_parts)

    return results


async def run_three_languages_eval(
    rollouts_path,
    output_path,
    concurrency=DEFAULT_CONCURRENCY,
    model=DEFAULT_JUDGE_MODEL,
    languages=None,
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

    languages = languages or DEFAULT_LANGUAGES
    logger.info("Evaluating languages: %s", languages)

    start_time = time.time()
    results = await evaluate_three_languages(
        rollouts,
        client,
        concurrency,
        logger,
        model,
        cost_tracker=tracker,
        languages=languages,
    )
    elapsed = time.time() - start_time

    save_results(results, output_path, logger)
    logger.info("Evaluation complete in %.1fs.", elapsed)
    tracker.log_summary(logger)
