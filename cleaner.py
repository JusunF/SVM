import pandas as pd
import re

df = pd.read_csv("data/raw/worldcup2014_tweets.csv")

def remove_url(text):
    if isinstance(text, str):
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    return text

# Apply the function to the tweet column
df["rawContent"] = df["rawContent"].apply(remove_url)

df.to_csv("worldcup2014_cleaned.csv", index=False)

print("URLs removed successfully!")