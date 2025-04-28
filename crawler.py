import gc
import heapq
import logging
import os
import pickle
from multiprocessing import Pool

from tqdm import tqdm

from corpus_interface import Document, DocumentAnnotation, UnifiedGetter
from document_rater import DocumentRater
from wandb_logger import WandbLogger

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


logger = logging.getLogger(__name__)


class Crawler:
    def __init__(
        self,
        unified_getter: UnifiedGetter,
        quality_raters: DocumentRater | list[DocumentRater],
        output_dir: str | None = None,
        num_workers: int = 1,
        wandb_logger: WandbLogger = None,
        max_num_in_mem_docs: int = 1000000,
    ) -> None:
        self.unified_getter = unified_getter
        self.output_dir = output_dir
        self.queue = []
        self.visited = set()
        self.quality_raters = (
            quality_raters if isinstance(quality_raters, list) else [quality_raters]
        )
        self.num_workers = num_workers
        self.wandb_logger = wandb_logger
        self.max_num_in_mem_docs = max_num_in_mem_docs
        self.require_doc_content = any(rater.require_doc_text() for rater in self.quality_raters)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.selected_doc_embeddings = []  # Stores embeddings of selected docs

    def put_into_queue(self, documents: list[Document]) -> None:
        for document in documents:
            heapq.heappush(self.queue, (document.annotations, document.docid))
            self.visited.add(document.docid)
        logger.info(f"Size after put: {len(self.queue)}")

    @staticmethod
    def _get_mean_score_for_logging(
        annotations: list[DocumentAnnotation], postfix: str
    ) -> dict[str, float]:
        result_dict = {}
        for key in annotations[0].keys():
            result_dict[f"mean_{key}_{postfix}"] = sum(
                annotation[key] for annotation in annotations
            ) / len(annotations)
        return result_dict

    def _log_all(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, float):
                logger.info(f"{key} = {value:.2f}")
            else:
                logger.info(f"{key} = {value}")
        if self.wandb_logger:
            self.wandb_logger.log(**kwargs)

    def pop_from_queue(self, num_docs: int) -> list[str]:
        if not self.queue:
            return []

        annotations, docids = [], []
        num_to_pop = min(num_docs, len(self.queue))
        for _ in range(num_to_pop):
            annotation, docid = heapq.heappop(self.queue)
            annotations.append(annotation)
            docids.append(docid)

        if annotations:
            mean_results = self._get_mean_score_for_logging(annotations, "pop")
            size_after_pop = len(self.queue)
            self._log_all(**mean_results, size_after_pop=size_after_pop)

        return docids


    def find_outinks(
        self, docids: list[str], with_predecessor_info: bool = False
    ) -> list[str] | list[tuple[str, list[str]]]:
        input_size = len(docids)
        with Pool(self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(self.unified_getter.get_outlinks, docids),
                    total=input_size,
                    desc="Finding outlinks",
                )
            )

        outlinks = []
        for result in results:
            outlinks.extend(result)
        if with_predecessor_info:
            outlinks_with_predecessor = []
            for docid, result in zip(docids, results, strict=True):
                outlinks_with_predecessor.append((docid, result))
        total_outlinks = len(outlinks)
        outlinks = set(outlinks)
        total_unique_outlinks = len(outlinks)
        outlinks -= self.visited
        total_unique_outlinks_unvisited = len(outlinks)
        self._log_all(
            total_outlinks=total_outlinks,
            total_unique_outlinks=total_unique_outlinks,
            total_unique_outlinks_unvisited=total_unique_outlinks_unvisited,
            expansion_ratio=total_outlinks / input_size,
            expansion_ratio_unique=total_unique_outlinks / input_size,
            expansion_ratio_unique_unvisited=total_unique_outlinks_unvisited / input_size,
            unvisited_ratio=total_unique_outlinks_unvisited / total_unique_outlinks,
        )
        outlinks = list(outlinks)
        return (outlinks, outlinks_with_predecessor) if with_predecessor_info else outlinks

    def _get_scores_for_docs(
        self, docids: list[str], current_partition: int, total_partitions: int
    ) -> list[Document]:
        if self.require_doc_content:
            with Pool(self.num_workers) as pool:
                all_docs = list(
                    tqdm(
                        pool.imap(self.unified_getter.get_doc, docids),
                        total=len(docids),
                        desc=f"Getting docs (partition {current_partition}/{total_partitions})",
                    )
                )
            all_docs = [doc for doc in all_docs if doc is not None]
        else:
            all_docs = [Document(docid=docid) for docid in docids]

        results = all_docs
        for quality_rater in self.quality_raters:
            results = quality_rater(results)

        #New: Apply semantic novelty boost
        # results = self._boost_scores_with_novelty(results)
        # self.selected_doc_embeddings.extend(
        #     self.embedding_model.encode([doc.text for doc in results], convert_to_numpy=True)
        # )

        if self.require_doc_content:
            for document in results:
                del document.text
        return results

    def get_scores_for_docs(self, docids: list[str]) -> list[Document]:
        logger.info(f"Getting scores for {len(docids)} docs")
        docids_partitions = [
            docids[i : i + self.max_num_in_mem_docs]
            for i in range(0, len(docids), self.max_num_in_mem_docs)
        ]
        for i, docids_partition in enumerate(docids_partitions):
            logger.info(f"Partition {i+1}: {len(docids_partition)} docs")
        results: list[Document] = []
        for i, docids_partition in enumerate(docids_partitions):
            results.extend(
                self._get_scores_for_docs(
                    docids_partition,
                    current_partition=i + 1,
                    total_partitions=len(docids_partitions),
                )
            )
        if self.require_doc_content:
            doc_hit_rate = len(results) / len(docids)
            self._log_all(doc_hit_rate=doc_hit_rate)
        annotations = [x.annotations for x in results]
        if not annotations:
            logger.warning("No annotations returned, skipping mean score logging.")
            return []

        mean_results = self._get_mean_score_for_logging(annotations, "push")
        self._log_all(**mean_results)
        return results
    

    def _boost_scores_with_novelty(self, docs: list[Document]) -> list[Document]:
        if not docs or not self.require_doc_content:
            return docs

        doc_texts = [doc.text for doc in docs]
        embeddings = self.embedding_model.encode(doc_texts, convert_to_numpy=True)
        
        if self.selected_doc_embeddings:
            selected_mean = np.mean(self.selected_doc_embeddings, axis=0)
            similarities = cosine_similarity(embeddings, [selected_mean]).flatten()
            novelty_factors = 1.0 - similarities  # Higher novelty = lower similarity

            for doc, boost in zip(docs, novelty_factors):
                for key in doc.annotations:
                    doc.annotations[key] *= (1.0 + boost)  # boost scores
        else:
            # No previous embeddings, store current and return as-is
            self.selected_doc_embeddings.extend(embeddings)

        return docs




    def write_output(self, iter_num: int, docids: list[str]) -> None:
        with open(os.path.join(self.output_dir, f"iter_{iter_num}.docids.txt"), "w") as fout:
            for docid in docids:
                fout.write(f"{docid}\n")


    def save_state(self, iter_num: int, num_selected_docs: int) -> None:
        state = {
            "queue": self.queue,
            "visited": self.visited,
            "num_selected_docs": num_selected_docs,
        }
        with open(os.path.join(self.output_dir, f"state_{iter_num:06d}.pkl"), "wb") as fout:
            pickle.dump(state, fout)

    def init_or_resume_state(self, state_file: str | None) -> tuple[int, int]:
        if state_file is None:
            logger.info("Starting from scratch")
            return 0, 0
        logger.info(f"Resuming from state file: {state_file}")
        iter_num = int(state_file.split("_")[-1].split(".")[0])
        with open(state_file, "rb") as fin:
            state = pickle.load(fin)
        self.queue, self.visited = state["queue"], state["visited"]
        num_selected_docs = state["num_selected_docs"]
        original_quality_raters = set(self.queue[0][0].keys())
        current_quality_raters = set(
            quality_rater.get_name() for quality_rater in self.quality_raters
        )
        gc.collect()
        if original_quality_raters != current_quality_raters:
            logger.info("Quality raters mismatch")
            logger.info(
                f"Quality raters in state file: {original_quality_raters}, "
                f"current quality raters: {current_quality_raters}"
            )
            logger.info("Current first item in the queue:")
            logger.info(self.queue[0])
            logger.info("Recomputing scores for all docs in the queue")
            recomputed = self.get_scores_for_docs([docid for _, docid in self.queue])
            assert len(recomputed) == len(self.queue)
            logger.info("Constructing new queue")
            new_queue = []
            for doc, (_, docid) in zip(recomputed, self.queue):
                assert doc.docid == docid
                new_queue.append((doc.annotations, docid))
            self.queue = new_queue
            heapq.heapify(self.queue)
            logger.info("After recomputation, first item in the queue:")
            logger.info(self.queue[0])
        return iter_num, num_selected_docs
