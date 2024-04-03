from abc import ABC, abstractmethod
from typing import List


class InputFeatures:
    def __init__(
        self,
        team_name_home: str,
        team_name_away: str,
        pts_home: int,
        pts_away: int,
        pos_home: float,
        pos_away: float,
        offensive_rating_home: float,
        defensive_rating_home: float,
        offensive_rating_away: float,
        defensive_rating_away: float,
        OR_home: float,
        DR_home: float,
        OR_away: float,
        DR_away: float,
        rest_home: int,
        rest_away: int,
    ) -> None:
        self.team_name_home = team_name_home
        self.team_name_away = team_name_away
        self.pts_home = pts_home
        self.pts_away = pts_away
        self.pos_home = pos_home
        self.pos_away = pos_away
        self.offensive_rating_home = offensive_rating_home
        self.defensive_rating_home = defensive_rating_home
        self.offensive_rating_away = offensive_rating_away
        self.defensive_rating_away = defensive_rating_away
        self.OR_home = OR_home
        self.DR_home = DR_home
        self.OR_away = OR_away
        self.DR_away = DR_away
        self.rest_home = rest_home
        self.rest_away = rest_away


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
