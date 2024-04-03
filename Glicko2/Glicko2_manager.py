from Glicko2.Glicko2_player import Glicko2_player
from Glicko2.Game_record import Game_record
from Glicko2.Glicko2_evaluator import glicko2_evaluate, glicko2_evaluate_no_complete

from typing import Dict, List, Optional
from math import e


def modified_sigmoid(x):
    return 1 / (1 + e ** (-x / 15))


class Glicko2_manager(object):
    def __init__(self) -> None:
        self.player_catalog: Optional[dict[str, Glicko2_player]] = None
        self.unprocess_game_record: list[Game_record] = []
        return

    """
    The Glicko-2 system works best when the number of games in a rating period is moderate to large, say an 
    average of at least 10-15 games per player in a rating period. 
    """

    def appraise_players(
        self, game_records=None
    ) -> Optional[Dict[str, Glicko2_player]]:
        if self.player_catalog is None:
            return None

        if game_records is not None:
            self.batch_load_game_records(game_records)

        updated_player_info = []
        for player_id, player_info in self.player_catalog.items():
            game_outcome = list(
                filter(
                    lambda game_record: game_record.contain_player(player_id),
                    self.unprocess_game_record,
                )
            )
            _, rating, deviation, volatility = player_info.unpack_values()

            if len(game_outcome) == 0:
                new_rating, new_deviation, new_volatility = (
                    glicko2_evaluate_no_complete(rating, deviation, volatility)
                )
            else:
                opponent_ids = list(
                    map(
                        lambda game_record: game_record.get_other_player(player_id),
                        game_outcome,
                    )
                )
                opponent_ratings = self.batch_get_rating(opponent_ids)
                opponent_deviations = self.batch_get_deviation(opponent_ids)
                outcomes = list(
                    map(
                        lambda game_record: game_record.get_player_result(player_id),
                        game_outcome,
                    )
                )

                # Applying the sigmoid function
                outcomes_transformed = list(map(modified_sigmoid, outcomes))

                new_rating, new_deviation, new_volatility = glicko2_evaluate(
                    rating,
                    deviation,
                    volatility,
                    opponent_ratings,
                    opponent_deviations,
                    outcomes_transformed,
                )

            updated_player_info.append(
                [player_id, new_rating, new_deviation, new_volatility]
            )

        for (
            player_id,
            player_rating,
            player_deviation,
            player_volatility,
        ) in updated_player_info:
            self.update_player_information(
                player_id, player_rating, player_deviation, player_volatility
            )

        self.unprocess_game_record = []
        return self.player_catalog

    def get_player_info(self, id) -> Glicko2_player:
        return self.player_catalog[id]

    def batch_get_rating(self, ids: List[str]) -> List[float]:
        return list(map(lambda id: self.player_catalog[id].rating, ids))

    def batch_get_deviation(self, ids: List[str]) -> List[float]:
        return list(map(lambda id: self.player_catalog[id].deviation, ids))

    def update_player_information(self, id, rating, deviation, volatility) -> bool:
        if id not in self.player_catalog:
            self.player_catalog[id] = Glicko2_player(id, rating, deviation, volatility)
            return False
        else:
            player_info = self.player_catalog[id]
            player_info.rating = rating
            player_info.deviation = deviation
            player_info.volatility = volatility
            return True

    def batch_load_player_data(self, player_data: Dict[str, List[float]]) -> None:
        if self.player_catalog is None:
            self.player_catalog = {}

        for player_id, (
            player_rating,
            player_deviation,
            player_volatility,
        ) in player_data.items():
            if player_id in self.player_catalog:
                continue
            self.player_catalog[player_id] = Glicko2_player(
                player_id, player_rating, player_deviation, player_volatility
            )

    def batch_load_game_records(self, game_records: List[Game_record]) -> None:
        self.unprocess_game_record.extend(game_records)

    def create_players_data(self, player_ids: List[str]) -> None:
        for player_id in player_ids:
            self.create_player_data(player_id)

    def create_player_data(self, player_id: str) -> None:
        if self.player_catalog is None:
            self.player_catalog = {}

        if player_id in self.player_catalog:
            return
        self.player_catalog[player_id] = Glicko2_player(player_id)

    def get_player_rating(self, id: str):
        return self.player_catalog[id].rating
