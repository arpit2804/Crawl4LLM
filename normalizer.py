from abc import ABC, abstractmethod

class ScoreNormalizer(ABC):
    _name: str | None = None

    @abstractmethod
    def __call__(self, score: float) -> float: ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def get_name(self) -> str | None:
        return self._name


class ZScoreNormalizer(ScoreNormalizer):
    _name = "zscore"

    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, score: float) -> float:
        if self.std == 0:
            return 0.0
        return (score - self.mean) / self.std


class MinMaxNormalizer(ScoreNormalizer):
    _name = "minmax"

    def __init__(self, min_score: float, max_score: float) -> None:
        self.min_score = min_score
        self.max_score = max_score

    def __call__(self, score: float) -> float:
        range_val = self.max_score - self.min_score
        if range_val == 0:
            return 0.0
        return (score - self.min_score) / range_val
