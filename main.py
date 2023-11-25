import numpy as np
import pandas as pd

import csv

# GLOBAL CONSTANT DECLARATIONS
NUM_TEAMS = 32 
NUM_STATS = 7

# 1. Database Initialization
def database_init():
    '''
    function: database_init

    returns all (selected) stats necessary for classification in dictionary {string: list}
    '''
    df_standard = pd.read_csv('all_data/standard_dataframe.csv')
    df_adv_goal = pd.read_csv('all_data/adv_goal_dataframe.csv')
    df_shooting = pd.read_csv('all_data/shooting_dataframe.csv')
    df_passing = pd.read_csv('all_data/passing_dataframe.csv')
    df_defensive = pd.read_csv('all_data/defensive_dataframe.csv')

    football_database = {}
    for idx in range(1, NUM_TEAMS+1):
        curr = [float(df_standard.iloc[idx]['Per 90 Minutes.9']), float(df_adv_goal.iloc[idx]['Expected.3']), 
                float(df_shooting.iloc[idx]['Standard.5']), float(df_passing.iloc[idx]['Short.2']), 
                float(df_passing.iloc[idx]['Medium.2']), float(df_passing.iloc[idx]['Long.2']),
                float(df_defensive.iloc[idx]['Challenges.2'])
                ]

        football_database[df_standard.iloc[idx]['Unnamed: 0_level_0']] = curr

    return football_database

# 2O. Training Dataset Initialization
def training_data_init(football_data):
    '''
    function: training_data_init

    returns training data necessary for training logistic regression (or more generally classification) models for UEFA
    Notes
         - [listofstats, winner]. listofstats = [team1, team2, allstats--->], winner=team1/team2 (via comparison)
    '''
    df_games = pd.read_csv('all_data/scores_fixtures_dataframe.csv')
    training_data = []

    idx = 1
    while df_games.iloc[idx]['Score'] != "NaN":
        pass
    return training_data

# 2B. BASELINE APPROACH OF MULTIPLICATION OF VALUES
def compiled_dataset(football_database, encodings):
    '''
    function: compiled_dataset

    encodings is positive or negative encodings indicating inverse of regular multiplication

    returns multiplied data
    '''
    baseline_data = {}
    for team_name, stats in football_database.items():
        curr = 1
        for idx, stat in enumerate(stats):
            curr += stat * encodings[idx] # 1 is positive stat, 0 is negative stat for encodings.
        baseline_data[team_name] = curr

    return baseline_data

# 3B. BASELINE APPROACH COMPARISON OF TEAMS


def main(type):
    football_database = database_init()
    if type == "baseline":
        # encoded positive or negative stats (multiplication for baseline)
        encodings = [1, 1, 1, 0.01, 0.01, 0.01, 0.01] # weights for linear combination
        baseline_data = compiled_dataset(football_database, encodings)
        baseline_data = sorted(baseline_data.items(), key=lambda x:x[1])
        print(baseline_data)
        for i in range(len(baseline_data)-1, 0, -1):
            print("Team Name: ", baseline_data[i][0], " Rating: ", baseline_data[i][1])
    elif type == "oracle":
        pass
    else:
        print("please either input 'baseline' or 'oracle' into the main function.")

if __name__ == "__main__":
    main("baseline")