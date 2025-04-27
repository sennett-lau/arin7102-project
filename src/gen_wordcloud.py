import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def drug_name(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-zA-Z\s]', '_', text)
    text = re.sub(r'\s+', '_', text)
    text = text.strip('_')
    return text

def generate_filename(title, filters, plot_type):
    """Generate a filename based on title and filters without redundancy"""
    # Extract base name (remove "Word Cloud for" or "Sentiment Analysis:" prefix)
    base_name = title.replace("Word Cloud for ", "").replace("Sentiment Analysis: ", "")
    
    # If there are filters, use only the filter values for the filename
    if filters:
        filter_str = "_".join([f"{k}_{drug_name(v)}" for k, v in filters.items()])
        sanitized_base = f"filtered_{filter_str}"
    else:
        sanitized_base = "all"
    
    # Combine components for the final filename
    return f"{sanitized_base}_{plot_type}.png"

def create_wordcloud(text_data, title, filters):
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         stopwords=set(stopwords.words('english')),
                         min_font_size=10, collocations=False).generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    
    # Save the word cloud
    filename = generate_filename(title, filters, "wordcloud")
    filename = f"./word clouds/" + filename
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved as: {filename}")

def apply_filters(df, filters):
    filtered_df = df.copy()
    for column, value in filters.items():
        filtered_df = filtered_df[filtered_df[column].str.lower() == value.lower()]
    return filtered_df

def analyze_sentiment(text_data):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for text in text_data:
        # Convert to string and handle non-string values
        text = str(text) if pd.notnull(text) else ""
        if text.strip():  # Only process non-empty strings
            score = sia.polarity_scores(text)
            sentiment_scores.append(score)
    
    # Return empty scores if no valid texts were processed
    if not sentiment_scores:
        return [{'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}]
    
    return sentiment_scores

def plot_sentiment(avg_sentiment, title, filters):
    sentiments = ['Negative', 'Neutral', 'Positive']
    values = [avg_sentiment['neg'], avg_sentiment['neu'], avg_sentiment['pos']]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sentiments, values, color=['red', 'gray', 'green'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title(f'Sentiment Analysis: {title}')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)
    
    # Save the sentiment plot
    filename = generate_filename(title, filters, "sentiment")
    filename = f"./word clouds/" + filename
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Sentiment plot saved as: {filename}")

def analyze_drugs(csv_file, filters=None):
    df = pd.read_csv(csv_file)
    
    if filters:
        df = apply_filters(df, filters)
        if len(df) == 0:
            print("No drug found with the specified filters")
            return
        title = "Word Cloud for Drugs with filters: " + ", ".join([f"{k}={v}" for k, v in filters.items()])
    else:
        title = "Word Cloud for All Drugs"
    
    all_drugs = ' '.join(df['Drug'].apply(drug_name))
    create_wordcloud(all_drugs, title, filters)

def analyze_reviews(csv_file, filters=None):
    df = pd.read_csv(csv_file)
    
    if filters:
        df = apply_filters(df, filters)
        if len(df) == 0:
            print("No reviews found with the specified filters")
            return
        title = "Word Cloud for Reviews with filters: " + ", ".join([f"{k}={v}" for k, v in filters.items()])
    else:
        title = "Word Cloud for All Reviews"
    
    all_reviews = ' '.join(df['Reviews'].apply(clean_text))
    create_wordcloud(all_reviews, title, filters)
    
    # Perform sentiment analysis
    sentiment_scores = analyze_sentiment(df['Reviews'])
    avg_sentiment = {
        'neg': sum(score['neg'] for score in sentiment_scores) / len(sentiment_scores),
        'neu': sum(score['neu'] for score in sentiment_scores) / len(sentiment_scores),
        'pos': sum(score['pos'] for score in sentiment_scores) / len(sentiment_scores),
        'compound': sum(score['compound'] for score in sentiment_scores) / len(sentiment_scores)
    }
    
    print("\nSentiment Analysis Results:")
    print(f"Average Negative Sentiment: {avg_sentiment['neg']:.2f}")
    print(f"Average Neutral Sentiment: {avg_sentiment['neu']:.2f}")
    print(f"Average Positive Sentiment: {avg_sentiment['pos']:.2f}")
    print(f"Average Compound Score: {avg_sentiment['compound']:.2f}")
    
    # Plot sentiment analysis results
    plot_sentiment(avg_sentiment, title, filters)
    
    # Interpret compound score
    if avg_sentiment['compound'] >= 0.05:
        print("Overall Sentiment: Positive")
    elif avg_sentiment['compound'] <= -0.05:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")

def get_available_columns(csv_file):
    df = pd.read_csv(csv_file)
    return list(df.columns)

def get_top_values(df, column, n=5):
    """Return the top n most frequent values for a given column"""
    return df[column].value_counts().head(n)

def main():
    csv_file = 'webmd.csv'
    available_columns = get_available_columns(csv_file)
    
    while True:
        print("\nWord Cloud Analysis Options:")
        print("1. Generate word cloud for drugs")
        print("2. Generate word cloud for reviews")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            filters = {}
            
            while True:
                print("\nAvailable columns to filter:", ", ".join(available_columns))
                column = input("Enter column name to filter (or 'done' to finish): ")
                if column.lower() == 'done':
                    break
                if column not in available_columns:
                    print("Invalid column name. Available columns:", ", ".join(available_columns))
                    continue
                
                # Show top 5 values for the selected column with numbered options
                df = pd.read_csv(csv_file)
                top_values = get_top_values(df, column)
                print(f"\nTop 5 most frequent values for {column}:")
                top_values_list = list(top_values.items())
                for i, (value, count) in enumerate(top_values_list, 1):
                    print(f"{i}. {value}: {count} occurrences")
                print("6. Enter a custom value")
                
                # Get user choice
                value_choice = input(f"\nEnter option (1-6) for {column}: ")
                try:
                    value_choice = int(value_choice)
                    if 1 <= value_choice <= 5:
                        # Select the corresponding value from top_values
                        value = top_values_list[value_choice - 1][0]
                    elif value_choice == 6:
                        # Allow custom input
                        value = input(f"Enter exact value for {column}: ")
                    else:
                        print("Invalid option. Please choose 1-6.")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 6.")
                    continue
                
                filters[column] = value
            
            if filters:
                analyze_drugs(csv_file, filters)
            else:
                print("No filters specified, generating word cloud for all drugs")
                analyze_drugs(csv_file)
        elif choice == '2':
            filters = {}
            
            while True:
                print("\nAvailable columns to filter:", ", ".join(available_columns))
                column = input("Enter column name to filter (or 'done' to finish): ")
                if column.lower() == 'done':
                    break
                if column not in available_columns:
                    print("Invalid column name. Available columns:", ", ".join(available_columns))
                    continue
                
                # Show top 5 values for the selected column with numbered options
                df = pd.read_csv(csv_file)
                top_values = get_top_values(df, column)
                print(f"\nTop 5 most frequent values for {column}:")
                top_values_list = list(top_values.items())
                for i, (value, count) in enumerate(top_values_list, 1):
                    print(f"{i}. {value}: {count} occurrences")
                print("6. Enter a custom value")
                
                # Get user choice
                value_choice = input(f"\nEnter option (1-6) for {column}: ")
                try:
                    value_choice = int(value_choice)
                    if 1 <= value_choice <= 5:
                        # Select the corresponding value from top_values
                        value = top_values_list[value_choice - 1][0]
                    elif value_choice == 6:
                        # Allow custom input
                        value = input(f"Enter exact value for {column}: ")
                    else:
                        print("Invalid option. Please choose 1-6.")
                        continue
                except ValueError:
                    print("Invalid input. Please enter a number between 1 and 6.")
                    continue
                
                filters[column] = value
            
            if filters:
                analyze_reviews(csv_file, filters)
            else:
                print("No filters specified, generating word cloud for all reviews")
                analyze_reviews(csv_file)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()