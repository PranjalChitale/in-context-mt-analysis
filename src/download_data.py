import argparse
from pathlib import Path

import datasets


def main(args):
    for split in ["dev", "test"]:
        path = Path(f"data/{split}")
        path.mkdir(parents=True, exist_ok=True)

    dataset_name_map = {
        "facebook/flores": "flores",
        "ai4bharat/IN22-Gen": "in22",
    }

    if args.dataset_name == "facebook/flores":
        splits = {"dev": "dev", "test": "devtest"}
    else:
        splits = {"test": "gen"}

    langs = args.langs
    for src_lang in langs:
        for tgt_lang in langs:
            if src_lang == tgt_lang:
                continue

            print(
                f"Downloading {src_lang}-{tgt_lang} from {args.dataset_name} ..."
            )

            try:
                dataset = datasets.load_dataset(
                    args.dataset_name,
                    f"{src_lang}-{tgt_lang}",
                    trust_remote_code=True,
                )
                for split in splits:
                    outfname = f"data/{split}/{dataset_name_map[args.dataset_name]}_{src_lang}-{tgt_lang}.jsonl"
                    dataset[splits[split]].to_json(outfname)
            except Exception as e:
                print(
                    f"Error downloading {src_lang}-{tgt_lang} from the {args.dataset_name}! See below the error trace:"
                )
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="facebook/flores",
        choices=["facebook/flores", "ai4bharat/IN22-Gen"],
        help="Name of the dataset from Huggingface Hub",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        type=str,
        help="Flores-200 codes indicating language and script",
    )
    args = parser.parse_args()

    main(args)
