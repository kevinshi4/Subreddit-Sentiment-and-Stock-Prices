import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch.nn.functional as F

# Load the pre-trained FinBERT model
model_name = "yiyanghkust/finbert-tone" #
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Initialize the sentiment analysis pipeline with truncation and max_length settings
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load the CSV file
input_file = './reddit_posts_with_stock_data.csv'
df = pd.read_csv(input_file)

# Ensure all text data is string and handle missing values
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')

# Apply the FinBERT model to label sentiment scores
def get_finbert_sentiment(text):
    # Truncate text properly using tokenizer's encoding capabilities
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    result = model(**inputs)
    logits = result.logits
    
    # Apply softmax to get probabilities
    probs = F.softmax(logits, dim=1)
    
    # Extract probabilities for each sentiment
    positive_prob = probs[0][0].item()
    negative_prob = probs[0][1].item()
    neutral_prob = probs[0][2].item()
    
    # Calculate sentiment score in the range -1 to 1
    sentiment_score = positive_prob - negative_prob
    
    return sentiment_score

df['finbert_sentiment_score'] = df['cleaned_text'].apply(get_finbert_sentiment)

# Save the results to a new CSV file
output_file = './reddit_posts_with_finbert_sentiment.csv'
df.to_csv(output_file, index=False)

print(f"Sentiment scores have been labeled and saved to {output_file}")
