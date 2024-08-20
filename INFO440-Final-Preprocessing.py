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

def get_top_posts(subreddit_name, top_limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    top_posts = []

    for submission in subreddit.top(time_filter='all', limit=top_limit):
        post_date = datetime.fromtimestamp(submission.created_utc)
        top_posts.append((submission.score, submission.title, submission.selftext, post_date))

    return top_posts

# Example usage for r/GameStop
top_posts_gamestop = get_top_posts('Gamestop', top_limit=1000)

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

# Calculate post volume (number of posts per day)
df_cleaned_posts['date'] = pd.to_datetime(df_cleaned_posts['date']).dt.date
post_volume = df_cleaned_posts.groupby('date').size().reset_index(name='post_volume')

# Merge post volume back into df_cleaned_posts
df_cleaned_posts = pd.merge(df_cleaned_posts, post_volume, on='date', how='left')

# Save data to CSV
file_path = 'gamestop_cleaned_reddit_posts.csv'
df_cleaned_posts.to_csv(file_path, index=False)
print(f"Data saved to {file_path}")


###################################
### Get Yahoo Finance Data Here ###
###################################

# Determine the start and end dates based on the posts data
start_date = df_cleaned_posts['date'].min().strftime('%Y-%m-%d')
end_date = df_cleaned_posts['date'].max().strftime('%Y-%m-%d')

print(f"Fetching stock data from {start_date} to {end_date}")

# Fetch stock data for Gamestop from Yahoo Finance
ticker_symbol = 'GME'  # GME ticker symbol
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Convert the stock_data index to a column for merging
stock_data.reset_index(inplace=True)

# Rename the 'Date' column to match the 'date' column in your Reddit data
stock_data.rename(columns={'Date': 'date'}, inplace=True)

# Ensure date columns are of the same type
df_cleaned_posts['date'] = pd.to_datetime(df_cleaned_posts['date']).dt.date
stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

# Merge the Reddit posts data with stock data on the 'date' column
df_combined = pd.merge(df_cleaned_posts, stock_data, on='date', how='inner')

# Debug: Check if merging was successful
print(f"Number of rows in combined data: {len(df_combined)}")
print(df_combined.head())

# Save the combined data to CSV
file_path_combined = 'gamestop_reddit_posts_with_stock_data.csv'
df_combined.to_csv(file_path_combined, index=False)
print(f"Combined data saved to {file_path_combined}")

###################################
### Smoothing Sentiment Scores ###
###################################

# Sort the DataFrame by date
df_combined = df_combined.sort_values('date')

# Convert the date column back to datetime for plotting
df_combined['date'] = pd.to_datetime(df_combined['date'])

# Calculate Weekly and Monthly Average Sentiment Score
df_combined['week'] = df_combined['date'].dt.to_period('W').dt.start_time
df_combined['month'] = df_combined['date'].dt.to_period('M').dt.start_time

df_weekly_avg = df_combined.groupby('week')['sentiment_score'].mean().reset_index()
df_monthly_avg = df_combined.groupby('month')['sentiment_score'].mean().reset_index()

# Function to plot background color based on sentiment
def plot_with_sentiment_background(ax, dates, sentiment_scores, color_map):
    for i in range(len(sentiment_scores)):
        color = 'green' if sentiment_scores[i] > 0 else 'red'
        ax.axvspan(dates[i], dates[i + 1] if i + 1 < len(dates) else dates[i], color=color, alpha=0.1)

# Plot Weekly Sentiment Background with Stock Close Price
fig, ax = plt.subplots(figsize=(12, 6))
plot_with_sentiment_background(ax, df_weekly_avg['week'], df_weekly_avg['sentiment_score'], 'coolwarm')
ax.plot(df_combined['date'], df_combined['Close'], label='Stock Close Price', color='blue')
ax.set_title('Weekly Sentiment vs Stock Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Close Price')
ax.grid(True)
plt.show()

# Plot Monthly Sentiment Background with Stock Close Price
fig, ax = plt.subplots(figsize=(12, 6))
plot_with_sentiment_background(ax, df_monthly_avg['month'], df_monthly_avg['sentiment_score'], 'coolwarm')
ax.plot(df_combined['date'], df_combined['Close'], label='Stock Close Price', color='blue')
ax.set_title('Monthly Sentiment vs Stock Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Close Price')
ax.grid(True)
plt.show()

# Plot Weekly Sentiment Background with Stock Volume
fig, ax = plt.subplots(figsize=(12, 6))
plot_with_sentiment_background(ax, df_weekly_avg['week'], df_weekly_avg['sentiment_score'], 'coolwarm')
ax.plot(df_combined['date'], df_combined['Volume'], label='Stock Volume', color='purple')
ax.set_title('Weekly Sentiment vs Stock Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Volume')
ax.grid(True)
plt.show()

# Plot Weekly Sentiment Background with Post Volume
fig, ax = plt.subplots(figsize=(12, 6))
plot_with_sentiment_background(ax, df_weekly_avg['week'], df_weekly_avg['sentiment_score'], 'coolwarm')
ax.plot(df_combined['date'], df_combined['post_volume'], label='Post Volume', color='red')
ax.set_title('Weekly Sentiment vs Post Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Post Volume')
ax.grid(True)
plt.show()

# Plot Monthly Sentiment Background with Stock Volume
fig, ax = plt.subplots(figsize=(12, 6))
plot_with_sentiment_background(ax, df_monthly_avg['month'], df_monthly_avg['sentiment_score'], 'coolwarm')
ax.plot(df_combined['date'], df_combined['Volume'], label='Stock Volume', color='purple')
ax.set_title('Monthly Sentiment vs Stock Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Volume')
ax.grid(True)
plt.show()

# Plot Monthly Sentiment Background with Post Volume
fig, ax = plt.subplots(figsize=(12, 6))
plot_with_sentiment_background(ax, df_monthly_avg['month'], df_monthly_avg['sentiment_score'], 'coolwarm')
ax.plot(df_combined['date'], df_combined['post_volume'], label='Post Volume', color='red')
ax.set_title('Monthly Sentiment vs Post Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Post Volume')
ax.grid(True)
