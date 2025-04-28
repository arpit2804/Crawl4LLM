import argparse
import glob
import os
import json
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from corpus_interface import ClueWeb22Api


def fetch(cw22_api: ClueWeb22Api, output_dir: str, input_file: str) -> None:
    basename_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{basename_without_ext}.jsonl")

    with open(input_file, "r") as f:
        doc_ids = [line.strip() for line in f]

    with open(output_file, "w") as f:
        for doc_id in doc_ids:
            doc = cw22_api.get_clean_text(doc_id)
            if doc:
                doc_stripped = doc.strip()
                if doc_stripped != "":
                    json.dump({"docid": doc_id, "text": doc_stripped}, f)
                    f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cw22_root_path", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = (
            f"{args.input_dir}_docs"
            if not args.input_dir.endswith("/")
            else f"{args.input_dir[:-1]}_docs"
        )

    if os.path.exists(output_dir):
        print(f"Output path {output_dir} already exists! Check again!")
        return
    os.makedirs(output_dir, exist_ok=True)

    cw22_api = ClueWeb22Api(args.cw22_root_path)

    all_input_files = glob.glob(os.path.join(args.input_dir, "*.txt"))
    all_input_files.sort()
    print(f"Number of files: {len(all_input_files)}")

    fetch_partial = partial(fetch, cw22_api, output_dir)
    with Pool(args.num_workers) as p:
        for _ in tqdm(p.imap(fetch_partial, all_input_files), total=len(all_input_files)):
            pass


if __name__ == "__main__":
    main()
