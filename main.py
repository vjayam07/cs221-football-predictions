import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from model import LogisticRegression
from torch.utils.data import TensorDataset, DataLoader

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
                float(df_shooting.iloc[idx]['Standard.5']), float(df_passing.iloc[idx]['Short.2'])*0.01, 
                float(df_passing.iloc[idx]['Medium.2'])*0.01, float(df_passing.iloc[idx]['Long.2'])*0.01,
                float(df_defensive.iloc[idx]['Challenges.2'])*0.01
                ]

        football_database[df_standard.iloc[idx]['Unnamed: 0_level_0']] = curr

    return football_database

# 2O. Training Dataset Initialization
def training_data_init(football_data):
    '''
    function: training_data_init

    returns training data necessary for training logistic regression (or more generally classification) models for UEFA
    Notes
         - (listofstats, winner). listofstats = [team1_name, team2_name, team1_data, team2_data], winner=0/1 (index) (via comparison)
         - 
    '''
    df_games = pd.read_csv('all_data/scores_fixtures_dataframe.csv')
    training_features = []
    training_outputs = []
    test_features = []
    test_outputs = []

    test_matches = []

    idx = 1
    while idx < df_games.shape[0]:
        team1 = df_games.iloc[idx]['Home']
        team2 = df_games.iloc[idx]['Away']

        xG_1 = float(df_games.iloc[idx]['xG'])
        xG_2 = float(df_games.iloc[idx]['xG.1'])

        if isinstance(df_games.iloc[idx]['Score'], float): 
            idx += 1
            continue
        score1 = float(df_games.iloc[idx]['Score'][:1])
        score2 = float(df_games.iloc[idx]['Score'][-1])

        if xG_1*np.sqrt(2)+score1*np.sqrt(7) > xG_2*np.sqrt(2)+score2*np.sqrt(7):
            winner = 0
        elif xG_1*np.sqrt(2)+score1*np.sqrt(7) == xG_2*np.sqrt(2)+score2*np.sqrt(7):
            winner = 0
        else:
            winner = 1 # tiebreak goes to team1

        arr = team1.split()
        team1 = arr[-1] + ' '
        for i, s in enumerate(arr):
            if i == len(arr)-2:
                team1 += s
                break
            team1 += s + ' '

        full_vec = np.concatenate((football_data[team1], football_data[team2]))
        if idx < 45:
            training_outputs.append(np.array([winner]))
            training_features.append(full_vec)
        else:
            test_outputs.append(np.array([winner]))
            test_features.append(full_vec)
            test_matches.append((team1, team2))
        idx += 1
        
    return training_features, training_outputs, test_features, test_outputs, test_matches

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
        for idx, stat in enumerate(stats[1]):
            curr += stat * encodings[idx]
        baseline_data[team_name] = curr

    return baseline_data

# 3O. NEW DATASET FOR UPCOMING MATCHES (STRICTLY FOR TESTING)
def new_matches(football_data):
    df_games = pd.read_csv('all_data/scores_fixtures_dataframe.csv')
    test_features = []
    test_matches = []

    idx = 0
    while idx < df_games.shape[0]:
        if df_games.iloc[idx]['Date'] == '2023-12-13' or df_games.iloc[idx]['Date'] == '2023-12-12': 
            team1 = df_games.iloc[idx]['Home']
            team2 = df_games.iloc[idx]['Away']
            
            arr = team1.split()
            team1 = arr[-1] + ' '
            for i, s in enumerate(arr):
                if i == len(arr)-2:
                    team1 += s
                    break
                team1 += s + ' '

            full_vec = np.concatenate((football_data[team1], football_data[team2]))
            test_features.append(full_vec)
            test_matches.append((team1, team2))
        idx += 1
    
    return test_features, test_matches


# 4O. TRAINING MODEL!
def train(training_features, training_outputs, validation_features=None, validation_outputs=None):
    features_tensor, outputs_tensor = torch.Tensor(training_features), torch.Tensor(training_outputs)
    dataset = TensorDataset(features_tensor, outputs_tensor)
    dataloader = DataLoader(dataset)

    # Model initialization
    input_size = features_tensor.shape[1]
    model = LogisticRegression(input_size)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print((epoch+1, loss.item()))
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
    return model

def test(model, test_features, test_outputs=None):
    # Assuming test_features and test_labels are your test data
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    all_predictions = np.array([])
    confidence = np.array([])
    if test_outputs:
        test_labels_tensor = torch.tensor(test_outputs, dtype=torch.float32).view(-1, 1)

        # Create a DataLoader for the test set
        test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=16)  # Adjust batch size as needed

        # Set model to evaluation mode
        model.eval()

        # Disable gradient calculations
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in test_loader:
                outputs = model(inputs)

                new_outputs = torch.flatten(outputs)
                confidence = np.concatenate((confidence, new_outputs.numpy()))

                predicted = (outputs > 0.5).float()  # Assuming a threshold of 0.5 for binary classification
                
                new_predicted = torch.flatten(predicted)
                all_predictions = np.concatenate((all_predictions, new_predicted.numpy()))
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()      
            accuracy = correct / total
            print(f'Accuracy of the model on the test set: {accuracy:.4f}')
    else:
        test_loader = DataLoader(test_features_tensor, batch_size=14)
        for inputs in test_loader:
                outputs = model(inputs)
                new_outputs = torch.flatten(outputs)
                confidence = np.concatenate((confidence, new_outputs.detach().numpy()))

                predicted = (outputs > 0.5).float()  # Assuming a threshold of 0.5 for binary classification
                new_predicted = torch.flatten(predicted)

                all_predictions = np.concatenate((all_predictions, new_predicted.numpy()))

    
    for idx, stat in enumerate(confidence):
        stat = np.abs(stat-0.5) / 0.5 * 100
        confidence[idx] = stat

    return all_predictions, confidence

def main(type):
    football_database = database_init()
    if type == "baseline":
        # encoded positive or negative stats (multiplication for baseline)
        encodings = [1, 1, 1, 1, 1, 1, 1] # weights for linear combination
        baseline_data = compiled_dataset(football_database, encodings)
        baseline_data = sorted(baseline_data.items(), key=lambda x:x[1])
        print(football_database)
        for i in range(len(baseline_data)-1, 0, -1):
            print("Team Name: ", baseline_data[i][0], " Rating: ", baseline_data[i][1])

    
    elif type == "oracle":
        training_features, training_outputs, test_features, test_outputs, test_matches = training_data_init(football_database)
        model = train(training_features, training_outputs)
        all_predictions, confidence = test(model, test_features, test_outputs=test_outputs)
        for idx, match in enumerate(test_matches):
            pred_winner = match[0] if all_predictions[idx] == 0 else match[1]
            true_winner = match[0] if test_outputs[idx] == 0 else match[1]
            print("Match ", idx+1 ,": ", match[0], " v.s. ", match[1])
            print("Predicted Winnner is ", pred_winner, " with ", confidence[idx], "% confidence.")
            print("True Winner is ", true_winner)
            if pred_winner == true_winner:
                print("Correctly Guessed!")
            print("#################################")
        
        print("")
        print("")
        
        new_features, upcoming_matches = new_matches(football_database)
        all_predictions, confidence = test(model, new_features)
        for idx, match in enumerate(upcoming_matches):
            pred_winner = match[0] if all_predictions[idx] == 0 else match[1]
            print("Match ", idx+1 ,": ", match[0], " v.s. ", match[1])
            print("Predicted Winnner is ", pred_winner, " with ", confidence[idx], "% confidence.")
            print("#################################")
            if idx == 15:
                break

    else:
        print("please either input 'baseline' or 'oracle' into the main function.")

if __name__ == "__main__":
    main("oracle")