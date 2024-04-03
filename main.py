from Models.GBR_Tree_Model import GBR_Tree_Model
from Models.Elo_only_model import Elo_Model
from Models.Average_Spread_Model import Average_Spread_Model
from Pipeline.Evaluate import evalute_model
from Pipeline.Data_anaylsis import generated_spread_histogram
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    TEAMS = {
        "Portland Trail Blazers",
        "Minnesota Timberwolves",
        "Washington Wizards",
        "Sacramento Kings",
        "Atlanta Hawks",
        "Orlando Magic",
        "Chicago Bulls",
        "San Antonio Spurs",
        "Cleveland Cavaliers",
        "Seattle SuperSonics",
        "Los Angeles Clippers",
        "Los Angeles Lakers",
        "Milwaukee Bucks",
        "Golden State Warriors",
        "Boston Celtics",
        "Vancouver Grizzlies",
        "Phoenix Suns",
        "Philadelphia 76ers",
        "Dallas Mavericks",
        "Miami Heat",
        "Houston Rockets",
        "New York Knicks",
        "Toronto Raptors",
        "Denver Nuggets",
        "New Jersey Nets",
        "Charlotte Hornets",
        "Utah Jazz",
        "Detroit Pistons",
        "Indiana Pacers",
        "Memphis Grizzlies",
        "New Orleans Hornets",
        "Charlotte Bobcats",
        "Oklahoma City Thunder",
        "New Orleans Pelicans",
        "Brooklyn Nets",
        "LA Clippers",
    }

    # elo_model = Elo_Model(TEAMS)
    # true_spread, predicted_spread = evalute_model(elo_model)

    # baseline_model = Average_Spread_Model()
    # true_spread, predicted_spread = evalute_model(baseline_model)

    tree_model = GBR_Tree_Model()
    true_spread, predicted_spread = evalute_model(tree_model)

    generated_spread_histogram(predicted_spread, true_spread)
