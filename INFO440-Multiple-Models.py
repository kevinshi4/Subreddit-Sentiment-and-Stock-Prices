import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from tqdm import tqdm

import warnings

# List of models to evaluate along with custom names for the sentiment score columns
model_configurations = [
    {"model_name": "yiyanghkust/finbert-tone", "custom_name": "FinBERT"},
    {"model_name": "finiteautomata/bertweet-base-sentiment-analysis", "custom_name": "BERTweet"},
    {"model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest", "custom_name": "RoBERTa"}
    # Add more models and custom names here if needed
]

# Load the CSV file
input_file = './reddit_posts_with_stock_data.csv'
df = pd.read_csv(input_file)

# Ensure all text data is string and handle missing values
df['cleaned_text'] = df['cleaned_text'].astype(str).fillna('')

# Define a function to calculate sentiment scores
def get_sentiment_score(model_name, text):
    # Load the tokenizer and model for the current model name using Auto classes
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Suppress specific warning related to weights not being used
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Truncate text properly using tokenizer's encoding capabilities
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    
    # Ensure inputs are on the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Perform the forward pass and handle potential issues
    with torch.no_grad():
        try:
            result = model(**inputs)
            logits = result.logits

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)

            # Extract probabilities for each sentiment
            positive_prob = probs[0][0].item()
            negative_prob = probs[0][1].item()

            # Calculate sentiment score in the range -1 to 1
            sentiment_score = positive_prob - negative_prob
        except IndexError as e:
            print(f"IndexError encountered for text: {text[:50]}... (truncated)")
            sentiment_score = 0  # Fallback to neutral score on error

    return sentiment_score

# Loop over each model configuration, calculate sentiment scores, and add them as columns
for config in model_configurations:
    model_name = config["model_name"]
    custom_name = config["custom_name"]

    # Apply the sentiment analysis function to the dataframe with a progress bar
    tqdm.pandas(desc=f"Processing with {custom_name}")
    df[f'{custom_name}_sentiment_score'] = df['cleaned_text'].progress_apply(lambda x: get_sentiment_score(model_name, x))

    print(f"Sentiment scores for model '{model_name}' have been labeled as '{custom_name}_sentiment_score'")



# Save all sentiment scores in a single CSV file
output_file = './reddit_posts_with_multiple_bert_sentiments.csv'
df.to_csv(output_file, index=False)

print(f"All sentiment scores have been saved to {output_file}")
