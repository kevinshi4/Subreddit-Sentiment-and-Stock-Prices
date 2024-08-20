"""
File to extract the data from Reddit
"""
import datetime
import itertools
import pandas as pd
import praw
import re
import time

import matplotlib.pyplot as plt
import nltk
import seaborn as sns

from datetime import datetime, timedelta
from nltk.corpus import stopwords
from praw.models import MoreComments
from tqdm import tqdm

from collections import Counter
from nltk.util import ngrams
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer


SECRET = *****
DEV = *****
APP = *****
APP_ID = *****


def authenticate():
    """
    Function to authenticate a Reddit PRAW connection
    :return: _reddit, an authenticated Reddit instance
    """
    _reddit = praw.Reddit(client_id=APP_ID,
                          client_secret=SECRET,
                          user_agent=DEV)
    return _reddit


def extract_posts(_reddit, _subreddit_name, _time_filter):
    """
    Function to extract top 10,000 posts from ‘r/____’ from the past year and save to a file
    :param _reddit: an authenticated Reddit instance
    :param _subreddit_name: a string, name of the subreddit
    :return: _posts_df, dataframe of subreddit
    """
    _top_posts = []

    for post in tqdm(_reddit.subreddit(_subreddit_name).top(time_filter=_time_filter, limit=10000), desc="Extracting posts"):
        _top_posts.append(post)

        # Stop if we have enough posts
        if len(_top_posts) >= 10000:
            break

        #time.sleep(1)  # Sleep to avoid hitting rate limits

    # create a dataframe of the posts
    _posts_df = pd.DataFrame([vars(post) for post in _top_posts])

    print(f"Extracted {len(_posts_df)} posts.")
    print(f"DataFrame shape: {_posts_df.shape}")

    # Save the DataFrame to a CSV file
    _posts_df.to_csv(f"reddit_data/{_subreddit_name}.csv", index=False)

    return _posts_df


def get_distribution(_df):
    """
    Function to get the distribution of extracted data
    :param _df: posts dataframe
    """
    # Convert the 'created_utc' column to datetime format
    _df['created_utc'] = pd.to_datetime(_df['created_utc'], unit='s')

    # Print out all the dates
    print(_df['created_utc'])
    # _df['created_utc'].to_csv('dates_output.csv', index=False)

    # Plotting the date distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(_df['created_utc'], bins=30, kde=True)  # kde=True adds a kernel density estimate
    plt.title('Date Distribution of Posts')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()

    earliest_date = _df['created_utc'].min()
    latest_date = _df['created_utc'].max()

    print(f"Earliest date: {earliest_date}")
    print(f"Latest date: {latest_date}")


if __name__ == "__main__":
    # authenticate a Reddit instance
    reddit_instance = authenticate()

    # get GameStop.csv dataframe
    try:
        gamestop_posts_df = pd.read_csv("reddit_data/GameStop.csv")
    except FileNotFoundError:
        gamestop_posts_df = extract_posts(reddit_instance, 'GameStop', _time_filter="year")
    print(gamestop_posts_df.head())
    print("GameStop.csv dataframe shape:", gamestop_posts_df.shape)
    print("GameStop.csv dataframe columns:", gamestop_posts_df.columns)
    get_distribution(gamestop_posts_df)

    # get Tesla.csv dataframe
    try:
        tesla_posts_df = pd.read_csv("reddit_data/Tesla.csv")
    except FileNotFoundError:
        tesla_posts_df = extract_posts(reddit_instance, 'Tesla', _time_filter="all")
    print(tesla_posts_df.head())
    print("Tesla.csv dataframe shape:", tesla_posts_df.shape)
    print("Tesla.csv dataframe columns:", tesla_posts_df.columns)
    get_distribution(tesla_posts_df)

    # get nvidia.csv dataframe
    try:
        nvidia_posts_df = pd.read_csv("reddit_data/nvidia.csv")
    except FileNotFoundError:
        nvidia_posts_df = extract_posts(reddit_instance, 'nvidia', _time_filter="year")
    print(nvidia_posts_df.head())
    print("nvidia.csv dataframe shape:", nvidia_posts_df.shape)
    print("nvidia.csv dataframe columns:", nvidia_posts_df.columns)
    get_distribution(nvidia_posts_df)