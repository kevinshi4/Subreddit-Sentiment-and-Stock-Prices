import praw
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
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

    for submission in subreddit.top(time_filter='all', limit=1000):  # Fetch top 1000 posts
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

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Combine title and text, remove special characters and URLs, convert to lowercase, and clean the text
def clean_text(title, selftext):
    # Combine title and body text
    combined_text = f"{title} {selftext}"

    # Remove URLs
    combined_text = re.sub(r'http\S+', '', combined_text)

    # Remove special characters, numbers, and punctuation
    combined_text = re.sub(r'[^A-Za-z\s]', '', combined_text)

    # Convert to lowercase
    combined_text = combined_text.lower()

    # Remove stopwords and lemmatize words
    cleaned_text = ' '.join(
        lemmatizer.lemmatize(word)
        for word in combined_text.split()
        if word not in stop_words
    )

    return cleaned_text


# Apply the cleaning function to your dataset
cleaned_posts = []
for score, title, selftext, date in top_posts_gamestop:
    cleaned_post = clean_text(title, selftext)
    cleaned_posts.append((score, cleaned_post, date))

df_cleaned_posts = pd.DataFrame(cleaned_posts, columns=['score', 'cleaned_text', 'date'])

# Save to example data to CSV
file_path = 'cleaned_reddit_posts.csv'
df_cleaned_posts.to_csv(file_path, index=False)
print(f"Data saved to {file_path}")


# Download the VADER lexicon and initialize sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Apply VADER sentiment analysis on each cleaned post
df_cleaned_posts['sentiment_score'] = df_cleaned_posts['cleaned_text'].apply(lambda text: sia.polarity_scores(text)['compound'])
print(df_cleaned_posts.head())

# Save the sentiment analysis to CSV
file_path_with_sentiment = 'cleaned_reddit_posts_with_sentiment.csv'
df_cleaned_posts.to_csv(file_path_with_sentiment, index=False)
print(f"Data with sentiment scores saved to {file_path_with_sentiment}")
