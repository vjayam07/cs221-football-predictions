# CS 221 Final Project: Scoring with AI! ⚽️

Welcome to Scoring with AI, Viraj, Rikhil, and Jacob's final project for CS 221. We attempt to predict the winners of [UEFA Champions' League](https://www.uefa.com/uefachampionsleague/) games through logistic regression.

## Dataset Sourcing and Webscraping

We extract all of our data from [FBREF](https://fbref.com/): a free, online database for football players, teams, and matches. To extract data, we use the `pandas` library - this extraction can be found in the `testing` folder of our codebase. Because FBREF has a cooldown against data scraping, we initialized our datasets one at a time in the `csv_init.py` file. 

We saved our data on a folder labeled `all_data`. Because this free repository could not sustain such memory, we locally trained our models on a Mac Mini computer and did not upload the processed `.csv` files. Here's a list of our initialized datasets as seen in `main.py`:

```
df_standard = pd.read_csv('all_data/standard_dataframe.csv')
df_adv_goal = pd.read_csv('all_data/adv_goal_dataframe.csv')
df_shooting = pd.read_csv('all_data/shooting_dataframe.csv')
df_passing = pd.read_csv('all_data/passing_dataframe.csv')
df_defensive = pd.read_csv('all_data/defensive_dataframe.csv')
df_games = pd.read_csv('all_data/scores_fixtures_dataframe.csv')
```

## Model Architecture

We used a simple linear regression model coded on PyTorch. A GPU is not needed to run this model. This is coded up in `model.py`. 

## Running the Model

To run the model, run `python main.py`. Take a look at the main block of the code:

```python
if __name__ == "__main__":
    main("oracle")
```

In the `main` function of the file, you will see that there are two choices for running the model: `oracle` or `baseline`. This indicates whether you would like to train a logistic regression model to train and test the given data or implement our baseline approach to obtain a ranking of all teams.
