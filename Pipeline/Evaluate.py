from typing import List
from Models.Model_interface import ModelInterface, InputFeatures
from Pipeline.Data_anaylsis import generated_spread_histogram
from Pipeline.Data_handler import load_game_data
from sklearn.metrics import mean_squared_error


# generate input features

"""
Assume that data pass in each method will be rows from the game.csv 
"""


def convert_data_into_InputFeatures(df) -> List[InputFeatures]:
    info = df[
        [
            "team_name_home",
            "team_name_away",
            "pts_home",
            "pts_away",
            "pos_home",
            "pos_away",
            "offensive_rating_home",
            "defensive_rating_home",
            "offensive_rating_away",
            "defensive_rating_away",
            "OR_home",
            "DR_home",
            "OR_away",
            "DR_away",
            "rest_home",
            "rest_away",
        ]
    ].values.tolist()

    processed_data = [
        InputFeatures(
            team_name_home,
            team_name_away,
            pts_home,
            pts_away,
            pos_home,
            pos_away,
            offensive_rating_home,
            defensive_rating_home,
            offensive_rating_away,
            defensive_rating_away,
            OR_home,
            DR_home,
            OR_away,
            DR_away,
            rest_home,
            rest_away,
        )
        for (
            team_name_home,
            team_name_away,
            pts_home,
            pts_away,
            pos_home,
            pos_away,
            offensive_rating_home,
            defensive_rating_home,
            offensive_rating_away,
            defensive_rating_away,
            OR_home,
            DR_home,
            OR_away,
            DR_away,
            rest_home,
            rest_away,
        ) in info
    ]
    return processed_data


def evalute_model(model: ModelInterface):
    game_df = load_game_data()
    total_rows = len(game_df)

    all_predicted_spread = []
    all_true_spread = []

    CROSS_VALIDATION = 50
    NUM_VALIDATION_GAME = 10
    for i in range(1, CROSS_VALIDATION):
        train_end_idx = i * total_rows // CROSS_VALIDATION
        training_df = game_df.iloc[:train_end_idx]

        validation_df = game_df.iloc[
            train_end_idx : train_end_idx + NUM_VALIDATION_GAME
        ]

        training_features = convert_data_into_InputFeatures(training_df)
        validation_features = convert_data_into_InputFeatures(validation_df)

        model.clear()
        model.train(training_features)
        predicted_spread = model.predict(validation_features)
        predicted_winner = [spread > 0 for spread in predicted_spread]

        # Realistly, we would be able to find the casino spreads
        true_spread = (validation_df["pts_home"] - validation_df["pts_away"]).values
        true_winner = (validation_df["pts_home"] > validation_df["pts_away"]).values

        accuary = sum(true_winner == predicted_winner) / len(predicted_winner)

        mse = mean_squared_error(true_spread, predicted_spread)
        print(
            f"Cross validation [{i}/{CROSS_VALIDATION - 1}]: MSE {mse: .1f}, Accurary: {accuary: .2f}"
        )
        all_predicted_spread.extend(predicted_spread)
        all_true_spread.extend(true_spread)

    return (all_true_spread, all_predicted_spread)
