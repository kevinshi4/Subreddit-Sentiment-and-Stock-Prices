
import praw
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Reddit API setup
SECRET = ""
DEV = ""
APP = ""
APP_ID = ""

# Reddit API credentials
reddit = praw.Reddit(client_id=APP_ID,
                     client_secret=SECRET,
                     user_agent=DEV)

# Calculate the timestamp for 2 years ago
two_years_ago = datetime.now() - timedelta(days=2*365)

def get_top_posts_last_two_years(subreddit_name, top_limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    top_posts = []

    for submission in subreddit.top(time_filter='all', limit=top_limit):  # Fetch top x posts
        post_date = datetime.fromtimestamp(submission.created_utc)
        if post_date >= two_years_ago:
            top_posts.append((submission.score, submission.title, submission.selftext, post_date))

    # Sort the posts by score in descending order
    top_posts = sorted(top_posts, key=lambda x: x[0], reverse=True)

    # Return the top X posts
    return top_posts[:top_limit]

# Example usage for r/GameStop
top_posts_gamestop = get_top_posts_last_two_years('GameStop', top_limit=10000)

# Print the results
for score, title, selftext, date in top_posts_gamestop:
    print(f"Score: {score}, Date: {date}")
    print(f"Title: {title}")
    print(f"Text: {selftext[:100]}...")  # Print the first 100 characters of the post
    print("-" * 80)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Combine title and text, remove special characters and URLs, convert to lowercase, and clean the text
def clean_text(title, selftext):
    combined_text = f"{title} {selftext}"
    combined_text = re.sub(r'http\S+', '', combined_text)
    combined_text = re.sub(r'[^A-Za-z\s]', '', combined_text)
    combined_text = combined_text.lower()
    cleaned_text = ' '.join(
        lemmatizer.lemmatize(word)
        for word in combined_text.split()
        if word not in stop_words
    )
    return cleaned_text

# Apply the cleaning function and sentiment analysis to your dataset
cleaned_posts = []
for score, title, selftext, date in top_posts_gamestop:
    cleaned_post = clean_text(title, selftext)
    sentiment_score = sia.polarity_scores(cleaned_post)['compound']
    cleaned_posts.append((score, cleaned_post, sentiment_score, date))

df_cleaned_posts = pd.DataFrame(cleaned_posts, columns=['score', 'cleaned_text', 'sentiment_score', 'date'])

# Save to example data to CSV
file_path = 'cleaned_reddit_posts.csv'
df_cleaned_posts.to_csv(file_path, index=False)
print(f"Data saved to {file_path}")


###################################
### Get Yahoo Finance Data Here ###
###################################

# Determine the start and end dates based on the posts data
start_date = df_cleaned_posts['date'].min().strftime('%Y-%m-%d')
end_date = df_cleaned_posts['date'].max().strftime('%Y-%m-%d')

print(f"Fetching stock data from {start_date} to {end_date}")

# Fetch stock data for GameStop from Yahoo Finance
ticker_symbol = 'GME'  # GameStop ticker symbol
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Debug: Check if stock data was fetched correctly
print(stock_data.head())
print(stock_data.tail())
print(f"Number of rows in stock data: {len(stock_data)}")

# Convert the stock_data index to a column for merging
stock_data.reset_index(inplace=True)

# Rename the 'Date' column to match the 'date' column in your Reddit data
stock_data.rename(columns={'Date': 'date'}, inplace=True)

# Ensure date columns are of the same type
df_cleaned_posts['date'] = pd.to_datetime(df_cleaned_posts['date']).dt.date
stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

# Merge the Reddit posts data with stock data on the 'date' column
df_combined = pd.merge(df_cleaned_posts, stock_data, on='date', how='inner')

# Display the combined DataFrame
print(df_combined.head())

# Save the combined data to CSV
file_path_combined = 'reddit_posts_with_stock_data.csv'
df_combined.to_csv(file_path_combined, index=False)
print(f"Combined data saved to {file_path_combined}")


import matplotlib.pyplot as plt

# Sort the DataFrame by date
df_combined = df_combined.sort_values('date')

# Convert the date column back to datetime for plotting
df_combined['date'] = pd.to_datetime(df_combined['date'])

# Create line plots for each variable

# Plot Sentiment Score over Time
plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['sentiment_score'], label='Sentiment Score')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Score Over Time')
plt.grid(True)
plt.show()

# Plot Reddit Post Score over Time
plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['score'], label='Reddit Post Score', color='orange')
plt.xlabel('Date')
plt.ylabel('Reddit Post Score')
plt.title('Reddit Post Score Over Time')
plt.grid(True)
plt.show()

# Plot Stock Close Price over Time
plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['Close'], label='Close Price', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Close Price Over Time')
plt.grid(True)
plt.show()

# Plot Stock Volume over Time
plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['Volume'], label='Volume', color='red')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Volume Over Time')
plt.grid(True)
plt.show()