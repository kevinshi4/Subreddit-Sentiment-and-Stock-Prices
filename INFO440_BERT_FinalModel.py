import praw
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def predict_sentiment(texts, model, tokenizer):
    model.eval()
    predictions = []
    for text in texts:
        encoding = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt')
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions.append(torch.argmax(outputs.logits, dim=1).item())
    return predictions


# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

def get_top_posts(subreddit_name, top_limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    top_posts = []

    # Fetch top X posts
    for submission in subreddit.top(time_filter='all', limit=top_limit):
        post_date = datetime.fromtimestamp(submission.created_utc)
        top_posts.append((submission.score, submission.title, submission.selftext, post_date))

    return top_posts

# Example usage for r/GameStop
top_posts_gamestop = get_top_posts('GameStop', top_limit=10000)

# Clean and prepare data
cleaned_posts = []
for score, title, selftext, date in top_posts_gamestop:
    cleaned_post = clean_text(title, selftext)
    cleaned_posts.append((score, cleaned_post, date))

df_cleaned_posts = pd.DataFrame(cleaned_posts, columns=['score', 'cleaned_text', 'date'])

# Tokenize the texts and create dataset
# NOTE: Replace with actual labels if available
dataset = SentimentDataset(df_cleaned_posts['cleaned_text'].tolist(), [0] * len(df_cleaned_posts), tokenizer)  # Dummy labels for demonstration

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

# Predict sentiment for cleaned posts
df_cleaned_posts['sentiment_score'] = predict_sentiment(df_cleaned_posts['cleaned_text'].tolist(), model, tokenizer)

# Fetch stock data and merge (as in your original code)
start_date = df_cleaned_posts['date'].min().strftime('%Y-%m-%d')
end_date = df_cleaned_posts['date'].max().strftime('%Y-%m-%d')

stock_data = yf.download('GME', start=start_date, end=end_date)
stock_data.reset_index(inplace=True)
stock_data.rename(columns={'Date': 'date'}, inplace=True)

df_cleaned_posts['date'] = pd.to_datetime(df_cleaned_posts['date']).dt.date
stock_data['date'] = pd.to_datetime(stock_data['date']).dt.date

df_combined = pd.merge(df_cleaned_posts, stock_data, on='date', how='inner')

# Save and plot
df_combined.to_csv('reddit_posts_with_stock_data.csv', index=False)
print(f"Combined data saved to 'reddit_posts_with_stock_data.csv'")

# Plotting code (as in your original script)
df_combined = df_combined.sort_values('date')
df_combined['date'] = pd.to_datetime(df_combined['date'])

plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['sentiment_score'], label='Sentiment Score')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Score Over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['score'], label='Reddit Post Score', color='orange')
plt.xlabel('Date')
plt.ylabel('Reddit Post Score')
plt.title('Reddit Post Score Over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['Close'], label='Close Price', color='green')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Close Price Over Time')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_combined['date'], df_combined['Volume'], label='Volume', color='red')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Stock Volume Over Time')
plt.grid(True)
plt.show()
