"""
vLLM-based rollout generator (high-throughput).

Used by `expert_iteration.py` for both the train-side rollouts under S1 and
the held-out test rollouts after SFT. vLLM gives much higher token throughput
than HF `model.generate()` thanks to paged attention + continuous batching,
so a full pass over thousands of prompts × N completions finishes in minutes
rather than hours.

Two modes:
  * No LoRA -> point vLLM directly at the base model checkpoint.
  * With LoRA -> load the base on CPU, merge the LoRA(s) via PEFT, save the
    merged model to a tempdir, then load that with vLLM. Tempdir is cleaned
    up after generation. We merge on CPU so the base + adapter footprint
    never lives on GPU concurrently with the vLLM engine.

The generator preserves the existing chat-template path (uses
`tokenizer.apply_chat_template(..., enable_thinking=False)` for Qwen) so the
prompts seen by the model match what HF generation would have produced.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import tempfile
import time
from typing import Optional


logger = logging.getLogger("rollouts_vllm")


def _format_prompts(messages_list, tokenizer, model_name: str = ""):
    chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
    if "qwen" in model_name.lower():
        chat_kwargs["enable_thinking"] = False
    return [
        tokenizer.apply_chat_template(msgs, **chat_kwargs)
        for msgs in messages_list
    ]


def _merge_lora_to_tempdir(
    base_model: str,
    lora_path: str,
    prior_lora_paths: Optional[list[str]],
    log: logging.Logger,
) -> str:
    """Merge LoRA(s) into the base model on CPU, save to a tempdir.

    Doing the merge on CPU keeps the base-weight footprint off the GPU so
    vLLM can claim the full device when it spins up.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("LoRA merge: loading base on CPU: %s", base_model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    for prior in prior_lora_paths or []:
        log.info("  applying prior LoRA: %s", prior)
        model = PeftModel.from_pretrained(model, prior, is_trainable=False)
        model = model.merge_and_unload()

    log.info("  applying LoRA: %s", lora_path)
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    tmpdir = tempfile.mkdtemp(prefix="vllm_merged_", dir="/workspace")
    log.info("  saving merged model -> %s", tmpdir)
    model.save_pretrained(tmpdir, safe_serialization=True)
    tokenizer.save_pretrained(tmpdir)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info("LoRA merge complete in %.1fs", time.time() - t0)
    return tmpdir


def generate_rollouts_vllm(
    *,
    model_path: str,
    messages_list: list[list[dict]],
    num_rollouts: int,
    gen_params: dict,
    lora_path: Optional[str] = None,
    prior_lora_paths: Optional[list[str]] = None,
    seed: int = 42,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    logger_: Optional[logging.Logger] = None,
) -> list[list[str]]:
    """Generate `num_rollouts` rollouts per prompt using vLLM.

    Returns list[P] of list[N] of decoded response strings, where P =
    len(messages_list) and N = num_rollouts. Order is preserved
    (responses[i] corresponds to messages_list[i]).
    """
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    log = logger_ or logger

    cleanup_tmpdir: Optional[str] = None
    if lora_path:
        merged_dir = _merge_lora_to_tempdir(
            model_path, lora_path, prior_lora_paths, log,
        )
        vllm_model_path = merged_dir
        cleanup_tmpdir = merged_dir
    else:
        vllm_model_path = model_path

    log.info("vLLM init: model=%s max_len=%d gpu_mem=%.2f",
             vllm_model_path, max_model_len, gpu_memory_utilization)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    try:
        llm = LLM(
            model=vllm_model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            seed=seed,
        )
    except Exception as e:
        log.warning(
            "vLLM init failed at gpu_mem=%.2f (%s); retrying at 0.5",
            gpu_memory_utilization, e,
        )
        llm = LLM(
            model=vllm_model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
            seed=seed,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        vllm_model_path, trust_remote_code=True
    )
    prompts = _format_prompts(
        messages_list, tokenizer, model_name=model_path,
    )

    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)

    sampling_params = SamplingParams(
        n=num_rollouts,
        temperature=gen_params.get("temperature", 1.0),
        top_p=gen_params.get("top_p", 0.95),
        top_k=gen_params.get("top_k", 50),
        max_tokens=gen_params.get("max_new_tokens", 1024),
        seed=seed,
        stop_token_ids=stop_token_ids or None,
    )

    log.info(
        "Generating: P=%d prompts x N=%d rollouts = %d total generations",
        len(prompts), num_rollouts, len(prompts) * num_rollouts,
    )
    log.info(
        "Sampling: temp=%.2f top_p=%.2f top_k=%d max_tokens=%d seed=%d",
        sampling_params.temperature, sampling_params.top_p,
        sampling_params.top_k, sampling_params.max_tokens, seed,
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    total = sum(len(o.outputs) for o in outputs)
    log.info(
        "vLLM done: %d generations in %.1fs (%.2f gen/s)",
        total, elapsed, total / max(elapsed, 1e-9),
    )

    responses: list[list[str]] = [[""] * num_rollouts for _ in range(len(prompts))]
    for i, output in enumerate(outputs):
        for j, comp in enumerate(output.outputs):
            responses[i][j] = comp.text

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("vLLM engine released")

    if cleanup_tmpdir and os.path.isdir(cleanup_tmpdir):
        log.info("Cleaning up merged-LoRA tempdir: %s", cleanup_tmpdir)
        shutil.rmtree(cleanup_tmpdir, ignore_errors=True)

    return responses
