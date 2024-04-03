from typing import List
from Models.Model_interface import InputFeatures, ModelInterface

import numpy as np
from collections import deque


class Average_Spread_Model(ModelInterface):
    NUM_GAME_AVERAGE = 3

    def __init__(self):
        self.game_history = {}

    def add_game(self, tag, spread):
        if tag not in self.game_history:
            self.game_history[tag] = deque()

        head_up_history = self.game_history[tag]
        head_up_history.append(spread)

        if len(head_up_history) > 3:
            head_up_history.popleft()

    def train(self, data: List[InputFeatures]):
        for game in data:
            spread = game.pts_home - game.pts_away

            tag = (game.team_name_home, game.team_name_away)
            reverse_tag = (game.team_name_away, game.team_name_home)

            self.add_game(tag, spread)
            self.add_game(reverse_tag, -spread)

    def predict(self, data: List[InputFeatures]):
        prediction_values = []
        for game in data:
            tag = (game.team_name_home, game.team_name_away)
            # If head to head game hasn't been played before, assume they are evenly matched
            if tag not in self.game_history:
                prediction_values.append(0)
                continue

            head_up_history = self.game_history[tag]
            predicted_spread = sum(head_up_history) / len(head_up_history)  # Average
            prediction_values.append(predicted_spread)
        return prediction_values

    def clear(self):
        self.game_history = {}
