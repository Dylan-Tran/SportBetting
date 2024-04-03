from typing import Tuple
from Glicko2.Glicko2_parameters import (
    INITIAL_RATING,
    INITIAL_DEVIATION,
    INITIAL_VOLATILITY,
)


class Glicko2_player(object):
    def __init__(
        self,
        id,
        rating=INITIAL_RATING,
        deviation=INITIAL_DEVIATION,
        volatility=INITIAL_VOLATILITY,
    ) -> None:
        self.id = id
        self.rating = rating
        self.deviation = deviation
        self.volatility = volatility

    def __hash__(self) -> int:
        return self.id.__hash__()

    def __repr__(self) -> str:
        return f"[id: {self.id}, rating: {self.rating}, deviation: {self.deviation}, volatility: {self.volatility}]\n"

    def unpack_values(self) -> Tuple[str, float, float, float]:
        return self.id, self.rating, self.deviation, self.volatility
