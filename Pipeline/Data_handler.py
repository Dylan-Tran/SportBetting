import pandas as pd
import os

# ELO and pipeline for models

DATA_CSV_PATH = "nba_data/csv/"
DATA_TYPES = [
    "common_player_info",
    "draft_combine_stats",
    "draft_history",
    "game",
    "game_info",
    "game_summary",
    "inactive_players",
    "line_score",
    "officials",
    "other_stats",
    # "play_by_play",
    "player",
    "team",
    "team_details",
    "team_history",
    "team_info_common",
]


def load_data():
    data_dict = {}
    for data_type in DATA_TYPES:
        data_dict[data_type] = pd.read_csv(f"{DATA_CSV_PATH}{data_type}.csv")
    return data_dict


def calculate_pos(fga, oreb, tov, fta):
    return (fga - oreb) + tov + 0.44 * fta


def calculate_rating_ema(df, window):
    df["offensive_rating_20_day_ma"] = (
        df["offensive_rating"].rolling(window=window, min_periods=1).mean().shift(1)
    )

    df["offensive_rating_20_day_ema"] = (
        df["offensive_rating"].ewm(span=window, adjust=True).mean().shift(1)
    )

    df["defensive_rating_20_day_ma"] = (
        df["defensive_rating"].rolling(window=window, min_periods=1).mean().shift(1)
    )

    df["defensive_rating_20_day_ema"] = (
        df["defensive_rating"].ewm(span=window, adjust=True).mean().shift(1)
    )


def load_game_data():
    df = pd.read_csv(f"{DATA_CSV_PATH}game.csv")
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df[df["game_date"].dt.year > 2012]
    df = df[df["season_type"] == "Regular Season"]
    df = df[df["game_id"] < 22100463]  # Not sure why this is needed

    df["pos_home"] = calculate_pos(
        df["fga_home"], df["oreb_home"], df["tov_home"], df["fta_home"]
    )
    df["pos_away"] = calculate_pos(
        df["fga_away"], df["oreb_away"], df["tov_away"], df["fta_away"]
    )

    df["offensive_rating_home"] = df["pts_home"] / df["pos_home"] * 100
    df["defensive_rating_home"] = df["pts_away"] / df["pos_away"] * 100

    df["offensive_rating_away"] = df["pts_away"] / df["pos_away"] * 100
    df["defensive_rating_away"] = df["pts_home"] / df["pos_home"] * 100

    df["spread_home"] = df["pts_home"] - df["pts_away"]

    WINDOW = 20
    df["offensive_rating_20_day_ma"] = (
        df["offensive_rating"].rolling(window=WINDOW, min_periods=1).mean().shift(1)
    )
    df["offensive_rating_20_day_ema"] = (
        df["offensive_rating"].ewm(span=WINDOW, adjust=True).mean().shift(1)
    )
    df["defensive_rating_20_day_ma"] = (
        df["defensive_rating"].rolling(window=WINDOW, min_periods=1).mean().shift(1)
    )
    df["defensive_rating_20_day_ema"] = (
        df["defensive_rating"].ewm(span=WINDOW, adjust=True).mean().shift(1)
    )

    return df
