'''
    file: win_loss_testing
'''
import pandas as pd
import numpy as np

df_games = pd.read_csv('all_data/scores_fixtures_dataframe.csv')

# football_database = {}
# for idx in range(1, NUM_TEAMS+1):
#     football_database[df_games.iloc[idx]['Unnamed: 0_level_0']] = curr

print(isinstance(df_games.iloc[90]['Score'], float))