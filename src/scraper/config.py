from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ACCOUNTS_DB = DATA_DIR / "accounts.db"
OUTPUT_CSV = RAW_DIR / "worldcup2014_tweets.csv"

# Search queries
SEARCH_QUERIES = [
    '"World Cup 2014" lang:en',
    '#WorldCup2014 lang:en',
    '#Brazil2014 lang:en',
]

# Maximum number of tweets to scrape for each query
LIMIT_PER_QUERY = 5000

# Date range (2014 FIFA World Cup)
DATE_SINCE = "2014-06-12"
DATE_UNTIL = "2014-07-14"

EXCLUDE_RETWEETS = True

# Columns to save in the CSV
CSV_COLUMNS = [
    "id",
    "rawContent",
    "date",
    "username",
    "likeCount",
    "retweetCount",
    "replyCount",
    "quoteCount",
    "viewCount",
    "lang",
    "hashtags",
    "url",
]