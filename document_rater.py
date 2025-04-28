import logging
import random
import re
from abc import abstractmethod
from functools import partial
from multiprocessing import Pool

import fasttext
from corpus_interface import Document, DocumentAnnotation, UnifiedGetter
from normalizer import ScoreNormalizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentRater:
    _name = None
    _require_doc_text = None

    def __init__(
        self, rater_name: str | None = None, normalizer: ScoreNormalizer | None = None, **kwargs
    ) -> None:
        self._obj_name = rater_name if rater_name is not None else self._name
        self.normalizer = normalizer
        logger.info(
            f"Initializing rater type: {self._name}, name: {self._obj_name}, normalizer: {self.normalizer}, args: {kwargs}"
        )

    def _annotate_doc(self, doc: Document, score: float | int) -> Document:
        if self.normalizer is None:
            return Document(
                docid=doc.docid,
                text=doc.text,
                annotations=DocumentAnnotation({self.get_name(): score, **doc.annotations}),
            )
        logger.debug(f"Normalizing score: {self.get_name()}")
        return Document(
            docid=doc.docid,
            text=doc.text,
            annotations=DocumentAnnotation(
                {
                    self.get_name(): score,
                    f"{self.get_name()}_{self.normalizer.get_name()}_normalized": self.normalizer(
                        score
                    ),
                    **doc.annotations,
                }
            ),
        )

    @abstractmethod
    def __call__(self, docs: list[Document]) -> list[Document]: ...

    def get_name(self) -> str | None:
        return self._obj_name

    @classmethod
    def require_doc_text(cls) -> bool | None:
        return cls._require_doc_text


class RandomRater(DocumentRater):
    _name = "random_score"
    _require_doc_text = False

    def __init__(
        self, rater_name: str | None = None, normalizer: ScoreNormalizer | None = None
    ) -> None:
        super().__init__(rater_name=rater_name, normalizer=normalizer)

    def __call__(self, docs: list[Document]) -> list[Document]:
        random_scores = [random.random() for _ in range(len(docs))]
        return [self._annotate_doc(doc, score) for doc, score in zip(docs, random_scores)]


class DocumentLengthRater(DocumentRater):
    _name = "length"
    _require_doc_text = True

    def __init__(
        self, rater_name: str | None = None, normalizer: ScoreNormalizer | None = None
    ) -> None:
        super().__init__(rater_name=rater_name, normalizer=normalizer)

    def __call__(self, docs: list[Document]) -> list[Document]:
        length_scores = [len(doc.text) for doc in docs]
        return [self._annotate_doc(doc, score) for doc, score in zip(docs, length_scores)]


class InlinkCountRater(DocumentRater):
    _name = "inlink_count"
    _require_doc_text = False

    def __init__(
        self,
        unified_getter: UnifiedGetter,
        rater_name: str | None = None,
        normalizer: ScoreNormalizer | None = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(
            rater_name=rater_name,
            unified_getter=unified_getter,
            normalizer=normalizer,
            num_workers=num_workers,
        )
        self.unified_getter = unified_getter
        self.num_workers = num_workers

    def _count_inlinks(self, docid: str) -> int:
        return len(self.unified_getter.get_inlinks(docid))

    def __call__(self, docs: list[Document]) -> list[Document]:
        if len(docs) <= 20000 or self.num_workers == 1:
            inlink_counts = [
                self._count_inlinks(doc.docid) for doc in tqdm(docs, desc="Counting inlinks")
            ]
        else:
            with Pool(self.num_workers) as pool:
                inlink_counts = list(
                    tqdm(
                        pool.imap(self._count_inlinks, [doc.docid for doc in docs]),
                        total=len(docs),
                        desc="Counting inlinks",
                    )
                )

        return [self._annotate_doc(doc, score) for doc, score in zip(docs, inlink_counts)]


_model_fasttext = None


def _init_fasttext_model(model_path: str) -> None:
    global _model_fasttext
    _model_fasttext = fasttext.load_model(model_path)


def normalize_text(text: str) -> str:
    text = re.sub(r"([.\!?,'/()])", r" \1 ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _predict_worker_fasttext(text: str, text_normalize: bool = False) -> float:
    global _model_fasttext
    text = " ".join(text.strip().splitlines())
    if text_normalize:
        text = normalize_text(text)
    pred = _model_fasttext.predict(text)
    (pred_label, pred_prob) = pred
    pred_label = pred_label[0]
    hq_prob = pred_prob[0]
    if pred_label == "__label__cc":
        hq_prob = 1 - hq_prob
    return hq_prob


class FasttextRater(DocumentRater):
    _name = "fasttext_score"
    _require_doc_text = True

    def __init__(
        self,
        model_path: str,
        rater_name: str | None = None,
        normalizer: ScoreNormalizer | None = None,
        text_normalize: bool = False,
        num_workers: int = 1,
    ) -> None:
        super().__init__(
            rater_name=rater_name,
            model_path=model_path,
            normalizer=normalizer,
            text_normalize=text_normalize,
            num_workers=num_workers,
        )
        self.model_path = model_path
        self.text_normalize = text_normalize
        self.num_workers = num_workers

    def _predict(self, text: str, text_normalize: bool = False) -> float:
        text = " ".join(text.strip().splitlines())
        if text_normalize:
            text = normalize_text(text)
        pred = self.model.predict(text)
        (pred_label, pred_prob) = pred
        pred_label = pred_label[0]
        hq_prob = pred_prob[0]
        if pred_label == "__label__cc":
            hq_prob = 1 - hq_prob
        return hq_prob

    def __call__(self, docs: list[Document]) -> list[Document]:
        results = []
        if len(docs) <= 100000 or self.num_workers == 1:
            # Load the model once in the main process
            self.model = fasttext.load_model(self.model_path)
            for doc in tqdm(docs, desc=f"Rating {self.get_name()}"):
                text: str = doc.text
                score = self._predict(text, self.text_normalize)
                results.append(self._annotate_doc(doc, score))
        else:
            predict_worker_fasttext_partial = partial(
                _predict_worker_fasttext, text_normalize=self.text_normalize
            )
            with Pool(
                self.num_workers, initializer=_init_fasttext_model, initargs=(self.model_path,)
            ) as pool:
                scores = list(
                    tqdm(
                        pool.imap(predict_worker_fasttext_partial, [doc.text for doc in docs]),
                        total=len(docs),
                        desc=f"Rating {self.get_name()}",
                    )
                )
            results = [self._annotate_doc(doc, score) for doc, score in zip(docs, scores)]
        return results


class EnsembleRater(DocumentRater):
    _name = "ensemble_score"
    _require_doc_text = False

    def __init__(
        self,
        raters_and_weights: list[dict],
        rater_name: str | None = None,
        normalizer: ScoreNormalizer | None = None,
    ) -> None:
        super().__init__(
            rater_name=rater_name, normalizer=normalizer, raters_and_weights=raters_and_weights
        )
        self.raters_and_weights = raters_and_weights

    def __call__(self, docs: list[Document]) -> list[Document]:
        scores = []
        for doc in tqdm(docs, desc=f"Rating {self.get_name()}"):
            total_score = 0
            for rater_and_weight in self.raters_and_weights:
                rater = rater_and_weight["rater_name"]
                weight = rater_and_weight["weight"]
                total_score += doc.annotations[rater] * weight
            scores.append(total_score)
        return [self._annotate_doc(doc, score) for doc, score in zip(docs, scores)]
