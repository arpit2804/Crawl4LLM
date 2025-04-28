import logging
import os

import numpy as np

from crawler import Crawler

logger = logging.getLogger(__name__)


def log_time(elapsed_time: int, remaining_time: int) -> None:
    h, rem = divmod(remaining_time, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        time_str = f"{int(h)}h {int(m)}m {int(s)}s"
    elif m > 0:
        time_str = f"{int(m)}m {int(s)}s"
    else:
        time_str = f"{int(s)}s"
    logger.info(f"Elapsed time this iter: {elapsed_time:.2f}s, remaining time: {time_str}")


def eval_and_plot(args, crawler: Crawler) -> None:
    import scipy.stats as stats
    from matplotlib import pyplot as plt

    logger.info("Initializing seed docs")
    if args.seed_docs_file is None:
        raise ValueError("Seed docs file must be provided")
    with open(args.seed_docs_file, "r") as fin:
        docids = [line.strip() for line in fin]

    crawler.wandb_logger = None
    results = crawler.get_scores_for_docs(docids)
    annotation_labels = args.plots
    num_raters = len(annotation_labels)

    fig, axes = plt.subplots(num_raters, num_raters, figsize=(18, 18))
    fig.suptitle("Correlations between different rating methods")

    for i, rater1 in enumerate(annotation_labels):
        for j, rater2 in enumerate(annotation_labels):
            ax = axes[i, j]
            try:
                r1_scores = [doc.annotations[rater1] for doc in results if rater1 in doc.annotations]
                r2_scores = [doc.annotations[rater2] for doc in results if rater2 in doc.annotations]
            except KeyError:
                logger.warning(f"Missing annotations for {rater1} or {rater2}")
                continue

            if i == j:
                ax.hist(r1_scores, bins=100, density=True, alpha=0.6, color="g")
                if "normalized" in rater1:
                    ax.set_xlim(-3, 3)
                ax.set_xlabel(rater1)
                ax.set_ylabel("Percentage (%)")
                ax.set_title(f"{rater1} Score Distribution")
            else:
                if len(r1_scores) == 0 or len(r2_scores) == 0:
                    logger.warning(f"No data for {rater1} vs {rater2}")
                    continue
                try:
                    corr, _ = stats.spearmanr(r1_scores, r2_scores)
                except ZeroDivisionError:
                    corr = 0.0
                logger.info(f"{rater1} vs {rater2}, spearman corr: {corr:.4f}")

                ax.scatter(r1_scores, r2_scores, label="Data points", alpha=0.5)

                m, b = np.polyfit(r1_scores, r2_scores, 1)
                ax.plot(r1_scores, np.array(r1_scores) * m + b, color="red", label="Fit line")

                bins = np.linspace(min(r1_scores), max(r1_scores), 100)
                bin_means = stats.binned_statistic(r1_scores, r2_scores, statistic="mean", bins=bins)[0]
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.plot(bin_centers, bin_means, color="blue", label="Mean values")

                ax.set_xlabel(rater1)
                ax.set_ylabel(rater2)
                if rater2 == "length":
                    ax.set_ylim(0, 30000)
                ax.set_title(f"Spearman corr: {corr:.2f}")
                ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(args.output_dir, "correlations.png")
    plt.savefig(output_path)
    plt.close()
