import argparse
import json
from typing import List, Tuple

import sacrebleu
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("langcode2name.json", "r", encoding="utf-8") as fp:
    language_mapping = json.load(fp)


def initialize_model_and_tokenizer(
    args: argparse.Namespace,
) -> Tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    """Initialize the model and tokenizer from the transformers library.

    Args:
        args (argparse.Namespace): Namespace object containing attributes:
            - tokenizer_name_or_path: Pretrained tokenizer name or path.
            - model_name_or_path: Pretrained model name or path.
            - device: Device to use for the model (e.g., "cpu" or "cuda")

    Returns:
        Tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
            A tuple containing pre-trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(args.device)
    model.eval()
    return model, tokenizer


def generate_completions(
    args: argparse.Namespace,
    batch_input_prompts: List[str],
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    chat_format: bool = False,
) -> List[str]:
    """Generate text completions from a Causal LM based on input prompts.

    Args:
        args (argparse.Namespace): Namespace object containing attributes:
            - device: Device to use for the model (e.g., "cpu" or "cuda")
        batch_input_prompts (List[str]): List of input prompts to be used for generations.
        model (transformers.AutoModelForCausalLM): Pre-trained Causal LM
        tokenizer (transformers.AutoTokenizer): Pre-trained tokenizer associated with Causal LM
        chat_format (bool, optional): Whether to apply chat template during tokenization (defaults to False).

    Returns:
        List[str]: List of generated text completions corresponding to each input prompt.
    """
    if chat_format:
        batch_input_prompts = [
            tokenizer.apply_chat_template(
                input_prompt, tokenize=False, add_generation_prompt=True
            )
            for input_prompt in batch_input_prompts
        ]

    encodings = tokenizer(
        batch_input_prompts,
        padding=True,
        return_tensors="pt",
        truncation=True,
    ).to(args.device)

    with torch.inference_mode():
        batch_outputs = model.generate(
            **encodings,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stop_id_sequences=None,
        )

    if chat_format:
        batch_input_prompts = [
            tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
            for prompt in batch_input_prompts
        ]

    batch_outputs = tokenizer.batch_decode(
        batch_outputs.detach().clone(), skip_special_tokens=True
    )
    batch_outputs = [
        output[len(prompt) :].strip().split("\n")[0]
        for prompt, output in zip(batch_input_prompts, batch_outputs)
    ]

    return batch_outputs


def compute_metrics(hypotheses, references):
    metrics = {
        "bleu": sacrebleu.corpus_bleu(
            hypotheses=hypotheses, references=[references], tokenize="flores200"
        ),
        "chrf": sacrebleu.corpus_chrf(
            hypotheses=hypotheses, references=[references]
        ),
        "chrf2": sacrebleu.corpus_chrf(
            hypotheses=hypotheses, references=[references], word_order=2
        ),
    }
    metrics = {k: v.score for k, v in metrics.items()}
    return metrics
