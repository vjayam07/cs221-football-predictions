import pandas as pd
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

df = pd.read_html('https://fbref.com/en/comps/8/schedule/Champions-League-Scores-and-Fixtures')
for idx,table in enumerate(df):
    print("***************************")
    print(idx)
    print(table)
