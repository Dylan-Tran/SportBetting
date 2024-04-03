import pandas as pd
import os


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

WINDOW = 20


def load_data():
    data_dict = {}
    for data_type in DATA_TYPES:
        data_dict[data_type] = pd.read_csv(f"{DATA_CSV_PATH}{data_type}.csv")
    return data_dict


def calculate_pos(fga, oreb, tov, fta):
    return (fga - oreb) + tov + 0.44 * fta


def calculate_ema(df_column):
    return (
        df_column.ewm(span=WINDOW, adjust=True).mean().reset_index(level=0, drop=True)
    )


def caclulate_ma(df_column):
    return (
        df_column.rolling(window=WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def generate_moving_average_features(df):
    # Remark the .loc is to guarantee that the orginal df is modified
    # HOME TEAM
    df.loc[:, "offensive_rating_20_day_ma_home"] = caclulate_ma(
        df.groupby("team_abbreviation_home")["offensive_rating_home"]
    )

    df.loc[:, "defensive_rating_20_day_ma_home"] = caclulate_ma(
        df.groupby("team_abbreviation_home")["defensive_rating_home"]
    )

    df.loc[:, "offensive_rating_20_day_ema_home"] = calculate_ema(
        df.groupby("team_abbreviation_home")["offensive_rating_home"]
    )

    df.loc[:, "defensive_rating_20_day_ema_home"] = calculate_ema(
        df.groupby("team_abbreviation_home")["defensive_rating_home"]
    )

    # AWAY TEAM
    df.loc[:, "offensive_rating_20_day_ma_away"] = caclulate_ma(
        df.groupby("team_abbreviation_away")["offensive_rating_away"]
    )

    df.loc[:, "defensive_rating_20_day_ma_away"] = caclulate_ma(
        df.groupby("team_abbreviation_away")["defensive_rating_away"]
    )

    df.loc[:, "offensive_rating_20_day_ema_away"] = calculate_ema(
        df.groupby("team_abbreviation_away")["offensive_rating_away"]
    )

    df.loc[:, "defensive_rating_20_day_ema_away"] = calculate_ema(
        df.groupby("team_abbreviation_away")["defensive_rating_away"]
    )


def generate_rest_features(df):
    df["rest_home"] = None
    df["rest_away"] = None
    temp_df = df.copy()

    for team in df["team_name_home"].unique():
        team_df = df[(df["team_name_home"] == team) | (df["team_name_away"] == team)]
        temp_df.loc[:, team] = team_df["game_date"].diff().dt.days

        mask_home = df["rest_home"].isnull()
        mask_away = df["rest_away"].isnull()

        df.loc[mask_home, "rest_home"] = temp_df.loc[
            temp_df["team_name_home"] == team, team
        ]

        df.loc[mask_away, "rest_away"] = temp_df.loc[
            temp_df["team_name_away"] == team, team
        ]


def load_game_data():
    df = pd.read_csv(f"{DATA_CSV_PATH}game.csv")
    df["game_date"] = pd.to_datetime(df["game_date"])

    df = df[df["game_date"].dt.year > 2012]
    df = df[df["season_type"] == "Regular Season"]
    df = df[df["game_id"] < 22100463]  # Not sure why this is needed

    # remove outlier games or games with a spread not between -50 and 50
    df = df[abs(df["pts_home"] - df["pts_away"]) < 50]

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

    generate_moving_average_features(df)

    # Eric wants this for some reasons? Why not use the columns directly
    df["OR_home"] = df["offensive_rating_20_day_ema_home"]
    df["DR_home"] = df["defensive_rating_20_day_ema_home"]

    df["OR_away"] = df["offensive_rating_20_day_ema_away"]
    df["DR_away"] = df["defensive_rating_20_day_ema_away"]

    generate_rest_features(df)

    df.dropna(inplace=True)

    return df
