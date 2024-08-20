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

    # Fetch top X posts
    for submission in subreddit.top(time_filter='all', limit=top_limit):
        post_date = datetime.fromtimestamp(submission.created_utc)
        top_posts.append((submission.score, submission.title, submission.selftext, post_date))

    return top_posts

# Example usage for r/GameStop
top_posts_gamestop = get_top_posts('GameStop', top_limit=1000)

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

# Debug: Check if Reddit data was fetched and processed correctly
print(f"Number of Reddit posts fetched: {len(df_cleaned_posts)}")

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

# Debug: Check date ranges and types
print(f"Reddit data date range: {df_cleaned_posts['date'].min()} to {df_cleaned_posts['date'].max()}")
print(f"Stock data date range: {stock_data['date'].min()} to {stock_data['date'].max()}")
print(f"Reddit data date type: {df_cleaned_posts['date'].dtype}")
print(f"Stock data date type: {stock_data['date'].dtype}")

# Merge the Reddit posts data with stock data on the 'date' column
df_combined = pd.merge(df_cleaned_posts, stock_data, on='date', how='inner')

# Debug: Check if merging was successful
print(f"Number of rows in combined data: {len(df_combined)}")
print(df_combined.head())

# Save the combined data to CSV
file_path_combined = 'gamestop_reddit_posts_with_stock_data.csv'
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

# Plot Post Volume over Time
plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['post_volume'], label='Post Volume', color='blue')
plt.xlabel('Date')
plt.ylabel('Post Volume')
plt.title('Post Volume Over Time')
plt.grid(True)
plt.show()



###################################
### Plotting Monthly Averages ###
###################################

# Convert the date column back to datetime for resampling
df_combined['date'] = pd.to_datetime(df_combined['date'])

# Resample by month and calculate the mean for numeric columns only
numeric_columns = df_combined.select_dtypes(include=['number']).columns
df_monthly_avg = df_combined.set_index('date')[numeric_columns].resample('M').mean().reset_index()

# Plot Sentiment Score Average per Month
plt.figure(figsize=(10, 6))
plt.plot(df_monthly_avg['date'], df_monthly_avg['sentiment_score'], label='Avg Sentiment Score per Month')
plt.xlabel('Month')
plt.ylabel('Avg Sentiment Score')
plt.title('Average Sentiment Score per Month')
plt.grid(True)
plt.show()

# Plot Reddit Post Score Average per Month
plt.figure(figsize=(10, 6))
plt.plot(df_monthly_avg['date'], df_monthly_avg['score'], label='Avg Reddit Post Score per Month', color='orange')
plt.xlabel('Month')
plt.ylabel('Avg Reddit Post Score')
plt.title('Average Reddit Post Score per Month')
plt.grid(True)
plt.show()

# Plot Stock Close Price Average per Month
plt.figure(figsize=(10, 6))
plt.plot(df_monthly_avg['date'], df_monthly_avg['Close'], label='Avg Stock Close Price per Month', color='green')
plt.xlabel('Month')
plt.ylabel('Avg Close Price')
plt.title('Average Stock Close Price per Month')
plt.grid(True)
plt.show()

# Plot Stock Volume Average per Month
plt.figure(figsize=(10, 6))
plt.plot(df_monthly_avg['date'], df_monthly_avg['Volume'], label='Avg Stock Volume per Month', color='red')
plt.xlabel('Month')
plt.ylabel('Avg Volume')
plt.title('Average Stock Volume per Month')
plt.grid(True)
plt.show()

# Plot Post Volume Average per Month
plt.figure(figsize=(10, 6))
plt.plot(df_monthly_avg['date'], df_monthly_avg['post_volume'], label='Avg Post Volume per Month', color='blue')
plt.xlabel('Month')
plt.ylabel('Avg Post Volume')
plt.title('Average Post Volume per Month')
plt.grid(True)
plt.show()
