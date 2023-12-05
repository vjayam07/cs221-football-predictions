# CS 221 Final Project: Scoring with AI! ⚽️

Welcome to Scoring with AI, Viraj, Rikhil, and Jacob's final project for CS 221. We attempt to predict the winners of [UEFA Champions' League](https://www.uefa.com/uefachampionsleague/) games through logistic regression.

## Dataset Sourcing and Webscraping

We extract all of our data from [FBREF](https://fbref.com/): a free, online database for football players, teams, and matches. To extract data, we use the `pandas` library - this extraction can be found in the `testing` folder of our codebase.

We locally saved our data on a folder labeled `all_data`. However, this free repository could not sustain such memory, so we locally trained our models on a Mac Mini computer.

## Running the Model

To run the model, run `python main.py`. In the 
