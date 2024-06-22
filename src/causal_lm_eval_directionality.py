import argparse
import os

import datasets
import pandas as pd
import torch
import transformers
import wandb
from tqdm import tqdm

from utils import (
    compute_metrics,
    generate_completions,
    initialize_model_and_tokenizer,
    language_mapping,
)


def compose_prompt(
    example: dict,
    src_lang: str,
    tgt_lang: str,
    devset: datasets.Dataset,
    k: int,
    seed: int,
    col_name: str = "prompt",
):
    src_col, tgt_col = f"sentence_{src_lang}", f"sentence_{tgt_lang}"

    prompt = f"Translate this from {language_mapping[src_lang]} into {language_mapping[tgt_lang]}:\n\n"  # noqa:E501

    if k > 0:
        # add few-shot in-context demonstrations
        demonstrations = devset.shuffle(seed=seed).select(range(k))
        for demonstration in demonstrations:
            prompt += f"{language_mapping[src_lang]}: {demonstration[src_col]}\n{language_mapping[tgt_lang]}: {demonstration[tgt_col]}\n\n"  # noqa:E501

    # add the test example
    prompt += f"{language_mapping[src_lang]}: {example[src_col]}\n{language_mapping[tgt_lang]}: "  # noqa:E501
    example[col_name] = prompt
    return example


def compose_chat_prompt(
    example: dict,
    src_lang: str,
    tgt_lang: str,
    devset: datasets.Dataset,
    k: int,
    seed: int,
    col_name: str = "prompt",
):
    src_col, tgt_col = f"sentence_{src_lang}", f"sentence_{tgt_lang}"

    messages = []
    messages.append(
        {
            "role": "system",
            "content": f"Translate this from {language_mapping[src_lang]} into {language_mapping[tgt_lang]}.\n\n",  # noqa:E501
        }
    )

    if k > 0:
        # add few-shot in-context demonstrations
        demonstrations = devset.shuffle(seed=seed).select(range(k))
        for demonstration in demonstrations:
            messages.append(
                {
                    "role": "user",
                    "content": f"{language_mapping[src_lang]}: {demonstration[src_col]}\n{language_mapping[tgt_lang]}: ",  # noqa:E501
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": f"{demonstration[tgt_col]}\n\n",
                }
            )

    # add the test example
    messages.append(
        {
            "role": "user",
            "content": f"{language_mapping[src_lang]}: {example[src_col]}\n{language_mapping[tgt_lang]}: ",  # noqa:E501
        }
    )
    example[col_name] = messages
    return example


def main(args):
    transformers.set_seed(args.seed)

    model_name = os.path.basename(args.model_name_or_path)
    run_name = f"{model_name}-{args.src_lang}-{args.tgt_lang}-{args.n_shot}shot"

    run = wandb.init(
        entity=args.wb_entity_name,
        project=args.wb_proj_name,
        name=run_name,
        save_code=True,
        config=vars(args),
    )

    print("loading data ...")
    dev_dataset = datasets.load_dataset("json", data_files=args.dev_fname)
    test_dataset = datasets.load_dataset("json", data_files=args.test_fname)
    dev_dataset, test_dataset = dev_dataset["train"], test_dataset["train"]

    dev_dataset = dev_dataset.remove_columns(
        [
            col
            for col in dev_dataset.column_names
            if col
            not in [f"sentence_{args.src_lang}", f"sentence_{args.tgt_lang}"]
        ]
    )
    test_dataset = test_dataset.remove_columns(
        [
            col
            for col in test_dataset.column_names
            if col
            not in [f"sentence_{args.src_lang}", f"sentence_{args.tgt_lang}"]
        ]
    )
    dataset = datasets.DatasetDict({"dev": dev_dataset, "test": test_dataset})

    if args.model_name_or_path in ["meta-llama/Llama-2-7b-chat-hf"]:
        dataset["test"] = dataset["test"].map(
            lambda x, i: compose_chat_prompt(
                x,
                args.src_lang,
                args.tgt_lang,
                dataset["dev"],
                args.n_shot,
                i,
            ),
            with_indices=True,
        )
    else:
        dataset["test"] = dataset["test"].map(
            lambda x, i: compose_prompt(
                x,
                args.src_lang,
                args.tgt_lang,
                dataset["dev"],
                args.n_shot,
                i,
            ),
            with_indices=True,
        )

    input_prompts = dataset["test"]["prompt"]
    print(f"Number of examples: {len(input_prompts)}\nHere are few examples: ")
    for i, input_prompt in enumerate(input_prompts[:5]):
        print(f"Example {i + 1}:")
        print("-" * 20)
        print(input_prompt)
        print()

    tgt_col = f"sentence_{args.tgt_lang}"

    print("Loading the model ...")
    model, tokenizer = initialize_model_and_tokenizer(args)
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, transformers.LlamaTokenizer) or isinstance(
        tokenizer, transformers.LlamaTokenizerFast
    ):
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    print("Generating the completions ...")
    hypotheses, references = [], []
    for start_idx in tqdm(range(0, len(input_prompts), args.batch_size)):
        end_idx = start_idx + args.batch_size
        batch_input_prompts = input_prompts[start_idx:end_idx]

        # only apply chat template for llama 2 chat model
        chat_format = args.model_name_or_path in [
            "meta-llama/Llama-2-7b-chat-hf"
        ]

        batch_hypotheses = generate_completions(
            args, batch_input_prompts, model, tokenizer, chat_format
        )
        batch_references = dataset["test"][tgt_col][start_idx:end_idx]

        hypotheses.extend(batch_hypotheses)
        references.extend(batch_references)

        if start_idx % 10 == 0:
            print(batch_hypotheses[0])

    print("Computing bleu, chrf, chrf++ ...")
    metrics = compute_metrics(hypotheses, references)

    if args.model_name_or_path in ["meta-llama/Llama-2-7b-chat-hf"]:
        predictions = [
            {
                "prompt": tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                ),
                "hypothesis": hypothesis,
                "reference": reference,
            }
            for prompt, hypothesis, reference in zip(
                input_prompts, hypotheses, references
            )
        ]
    else:
        predictions = [
            {"prompt": prompt, "hypothesis": hypothesis, "reference": reference}
            for prompt, hypothesis, reference in zip(
                input_prompts, hypotheses, references
            )
        ]

    pred_table = wandb.Table(dataframe=pd.DataFrame(predictions))
    run.log({"predictions": pred_table, **metrics})

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Name or path of the pre-trained language model.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Name or path of the tokenizer associated with the language model.",
    )
    parser.add_argument(
        "--test_fname",
        type=str,
        default="data/test/flores_eng_Latn-hin_Deva.jsonl",
        help="Name or path of the test dataset for evaluation.",
    )
    parser.add_argument(
        "--dev_fname",
        type=str,
        default="data/dev/flores_eng_Latn-hin_Deva.jsonl",
        help="Name or path of the development dataset for fine-tuning or validation.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory to cache the pre-trained language model and tokenizer.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source language code (e.g., eng_Latn for English in Latin script).",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="Target language code (e.g., hin_Deva for Hindi in Devanagari script).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device type where the model should be loaded for inference.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed value for reproducibility."
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=4,
        help="Number of in-context demonstrations for prompting.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for text generation and evaluation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum numbers of tokens to generate, ignoring the number of tokens in the prompt during text generation.",
    )
    parser.add_argument(
        "--wb_entity_name",
        type=str,
        default="llm-icl",
        help="Entity name for logging experiment on wandb.",
    )
    parser.add_argument(
        "--wb_proj_name",
        type=str,
        default="directionality",
        help="Project name for logging experiment on wandb.",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    assert args.test_fname.endswith(".jsonl") and args.dev_fname.endswith(
        ".jsonl"
    ), "test and dev files should be jsonl."

    assert (
        args.device == "cuda" and torch.cuda.is_available()
    ), "No GPU device available for experiment."

    main(args)
