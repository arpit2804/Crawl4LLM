import argparse
import json
import logging
import os
import random
import time
from collections import deque

import numpy as np
import yaml

from corpus_interface import ClueWeb22Api, DocumentAnnotation, UnifiedGetter
from crawler import Crawler
from document_rater import (
    DocumentLengthRater,
    DocumentRater,
    EnsembleRater,
    FasttextRater,
    InlinkCountRater,
    RandomRater,
)
from normalizer import MinMaxNormalizer, ZScoreNormalizer
from utils import eval_and_plot, log_time
from wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


def initialize_quality_raters(args, unified_getter: UnifiedGetter) -> list[DocumentRater]:
    quality_raters = []
    for rating_method in args.rating_methods:
        type_ = rating_method["type"]
        other_args = {k: v for k, v in rating_method.items() if k != "type"}
        if "num_workers" not in other_args:
            other_args["num_workers"] = args.num_workers
        normalizer = None
        if "normalizer" in other_args:
            normalizer_type = other_args["normalizer"]["type"]
            normalizer_args = {k: v for k, v in other_args["normalizer"].items() if k != "type"}
            if normalizer_type == "zscore":
                normalizer = ZScoreNormalizer(**normalizer_args)
            elif normalizer_type == "minmax":
                normalizer = MinMaxNormalizer(**normalizer_args)
            else:
                raise ValueError(f"Unknown normalizer type: {normalizer_type}")
            other_args.pop("normalizer")
        match type_:
            case "random_score":
                quality_raters.append(RandomRater(normalizer=normalizer))
            case "length":
                quality_raters.append(DocumentLengthRater(normalizer=normalizer))
            case "inlink_count":
                quality_raters.append(InlinkCountRater(unified_getter=unified_getter, normalizer=normalizer, **other_args))
            case "fasttext_score":
                quality_raters.append(FasttextRater(normalizer=normalizer, **other_args))
            case "ensemble_score":
                pass
            case _:
                raise ValueError(f"Unknown rating method: {type_}")
    for rating_method in args.rating_methods:
        if rating_method["type"] == "ensemble_score":
            other_args = {k: v for k, v in rating_method.items() if k != "type"}
            quality_raters.append(EnsembleRater(**other_args))
            break
    return quality_raters


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["crawl", "rate"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cw22_root_path", type=str)
    parser.add_argument("--seed_docs_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--num_selected_docs_per_iter", type=int, default=10000)
    parser.add_argument("--max_num_docs", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_state_every", type=int, default=400)
    parser.add_argument("--resume_from_state", type=str, default=None)
    parser.add_argument("--max_num_in_mem_docs", type=int, default=1000000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args, _ = parser.parse_known_args()

    config_file = args.config
    with open(config_file, "r") as fin:
        if config_file.endswith(".yaml") or config_file.endswith(".yml"):
            config = yaml.safe_load(fin)
        elif config_file.endswith(".json"):
            config = json.load(fin)
        else:
            raise ValueError("Config file must be either yaml or json")
        rating_methods = config["rating_methods"]
        plots = config.get("plots", [])
    config = {k: v for k, v in config.items() if k != "rating_methods"}
    parser.set_defaults(**config)
    args = parser.parse_args()
    args.rating_methods = rating_methods
    args.plots = plots

    return args


def main():
    args = parse_arguments()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    wandb_logger = WandbLogger(args.wandb_project, args.wandb_run_name, args) if args.wandb else None
    logger.info(f"Number of workers: {args.num_workers}")

    DocumentAnnotation.set_compare_method(args.selection_method, args.order)

    cw22_api = ClueWeb22Api(args.cw22_root_path)
    unified_getter = UnifiedGetter(cw22_api=cw22_api)
    quality_raters = initialize_quality_raters(args, unified_getter)
    crawler = Crawler(
        unified_getter=unified_getter,
        quality_raters=quality_raters,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        wandb_logger=wandb_logger,
        max_num_in_mem_docs=args.max_num_in_mem_docs,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "rate":
        eval_and_plot(args, crawler)
        return

    logger.info(f"Target number of docs: {args.max_num_docs}")
    iter_num, num_selected_docs = crawler.init_or_resume_state(args.resume_from_state)
    logger.info(f"Starting from iteration {iter_num}, total selected docs: {num_selected_docs}")

    if iter_num == 0:
        logger.info("Initializing seed docs")
        if args.seed_docs_file is None:
            raise ValueError("Seed docs file must be provided")
        with open(args.seed_docs_file, "r") as fin:
            seed_docids = [line.strip() for line in fin]
        assert len(seed_docids) >= args.num_selected_docs_per_iter, f"Insufficient seed docs: {len(seed_docids)}"
        logger.info(f"Sampling {args.num_selected_docs_per_iter} seed docs")
        seed_docids = random.sample(seed_docids, args.num_selected_docs_per_iter)
        print("[Arpit] this is the first check")
        crawler.put_into_queue(crawler.get_scores_for_docs(seed_docids))
        print("[Arpit] this is the second check")

    if wandb_logger:
        wandb_logger.step()

    elapsed_time_each_iter = deque(maxlen=10)
    while True:
        time_start = time.time()
        logger.info(f"ITERATION {iter_num}")
        if not crawler.queue:
            break
        docids = crawler.pop_from_queue(args.num_selected_docs_per_iter)
        num_selected_docs += len(docids)
        logger.info(
            f"Number of selected docs in this iter: {len(docids)}, total: {num_selected_docs} ({num_selected_docs/args.max_num_docs:.2%})"
        )
        if wandb_logger:
            wandb_logger.log(crawled_docs=num_selected_docs)
        crawler.write_output(iter_num, docids)

        if num_selected_docs >= args.max_num_docs:
            logger.info("Reached target number of docs, stopping")
            logger.info("Saving final state")
            crawler.save_state(iter_num + 1, num_selected_docs)
            break

        outlinks = crawler.find_outinks(docids)
        if outlinks:
            scores = crawler.get_scores_for_docs(outlinks)
            if scores:
                crawler.put_into_queue(scores)
        iter_num += 1

        elapsed_time = time.time() - time_start
        elapsed_time_each_iter.append(elapsed_time)
        mean_elapsed_time = sum(elapsed_time_each_iter) / len(elapsed_time_each_iter)
        remaining_time = mean_elapsed_time * (args.max_num_docs - num_selected_docs) / args.num_selected_docs_per_iter
        log_time(elapsed_time, remaining_time)

        if wandb_logger:
            wandb_logger.log(elapsed_time=elapsed_time).step()

        if args.save_state_every > 0 and iter_num % args.save_state_every == 0:
            logger.info(f"Saving state at iteration {iter_num}")
            crawler.save_state(iter_num, num_selected_docs)


if __name__ == "__main__":
    main()