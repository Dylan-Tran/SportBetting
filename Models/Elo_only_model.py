from typing import List
from Models.Model_interface import InputFeatures, ModelInterface
from Glicko2.Glicko2_manager import Glicko2_manager
from Glicko2.Game_record import Game_record

import numpy as np


class Elo_Model(ModelInterface):
    WARM_UP_PERIOD = 100

    def __init__(self, teams):
        self.teams = teams
        self.elo_manager = Glicko2_manager()
        self.elo_manager.create_players_data(teams)
        self.prediction_function = None
        self.num_games_seen = 0

        self.elo_differences_score_differences: List[List[float]] = []

    def train(self, data: List[InputFeatures]):
        game_history: List[Game_record] = []
        for game in data:
            spread = game.pts_home - game.pts_away
            game_record = Game_record(game.team_name_home, game.team_name_away, spread)
            game_history.append(game_record)

        BLOCK_SIZE = 4
        for i in range(len(game_history) // BLOCK_SIZE):
            elo_training_block = game_history[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            regular_training_block = data[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            self.elo_manager.appraise_players(elo_training_block)
            self.num_games_seen += BLOCK_SIZE

            if self.num_games_seen >= Elo_Model.WARM_UP_PERIOD:
                for game in regular_training_block:
                    spread = game.pts_home - game.pts_away
                    home_team_rating, away_team_rating = (
                        self.elo_manager.batch_get_rating(
                            [game.team_name_home, game.team_name_away]
                        )
                    )

                    elo_differences = home_team_rating - away_team_rating
                    self.elo_differences_score_differences.append(
                        [elo_differences, spread]
                    )

        self.elo_manager.appraise_players(game_history)
        self.prediction_function = None

    def generate_prediction_function(self) -> None:
        x_points = list(map(lambda x: x[0], self.elo_differences_score_differences))
        y_points = list(map(lambda x: x[1], self.elo_differences_score_differences))

        # Find best fit line
        m, b = np.polyfit(x_points, y_points, 1)

        return lambda elo_diff: (m * elo_diff + b)

    def predict(self, data: List[InputFeatures]):
        if self.prediction_function is None:
            self.prediction_function = self.generate_prediction_function()

        prediction_values = []
        for game in data:
            home_team_rating = self.elo_manager.get_player_rating(game.team_name_home)
            away_team_rating = self.elo_manager.get_player_rating(game.team_name_away)
            prediction_values.append(
                self.prediction_function(home_team_rating - away_team_rating)
            )

        return prediction_values

    def clear(self):
        self.elo_manager = Glicko2_manager()
        self.elo_manager.create_players_data(self.teams)
        self.prediction_function = None
        self.num_games_seen = 0
        self.elo_differences_score_differences: List[List[float]] = []
