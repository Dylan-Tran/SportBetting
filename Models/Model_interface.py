from abc import ABC, abstractmethod
from typing import List


class InputFeatures:
    def __init__(
        self, team_name_home: str, team_name_away: str, pts_home: int, pts_away: int
    ) -> None:
        self.team_name_home = team_name_home
        self.team_name_away = team_name_away
        self.pts_home = pts_home
        self.pts_away = pts_away


class ModelInterface(ABC):
    @abstractmethod
    def train(self, data: List[InputFeatures]):
        pass

    @abstractmethod
    def predict(self, data: List[InputFeatures]):
        pass

    @abstractmethod
    def clear(self):
        pass
