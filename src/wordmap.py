import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('vader_lexicon')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def create_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         stopwords=set(stopwords.words('english')),
                         min_font_size=10).generate(text_data)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

def apply_filters(df, filters):
    filtered_df = df.copy()
    for column, value in filters.items():
        filtered_df = filtered_df[filtered_df[column].str.lower() == value.lower()]
    return filtered_df

def analyze_sentiment(text_data):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for text in text_data:
        score = sia.polarity_scores(text)
        sentiment_scores.append(score)
    
    return sentiment_scores

def plot_sentiment(avg_sentiment, title):
    # Prepare data for plotting
    sentiments = ['Negative', 'Neutral', 'Positive']
    values = [avg_sentiment['neg'], avg_sentiment['neu'], avg_sentiment['pos']]
    
    # Create bar plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(sentiments, values, color=['red', 'gray', 'green'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title(f'Sentiment Analysis: {title}')
    plt.ylabel('Average Score')
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
    plt.show()

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
    create_wordcloud(all_reviews, title)
    
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
    plot_sentiment(avg_sentiment, title)
    
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
        print("1. Generate word cloud for all reviews")
        print("2. Generate word cloud with custom filters")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            analyze_reviews(csv_file)
        elif choice == '2':
            filters = {}
            print("\nAvailable columns to filter:", ", ".join(available_columns))
            
            while True:
                column = input("Enter column name to filter (or 'done' to finish): ")
                if column.lower() == 'done':
                    break
                if column not in available_columns:
                    print("Invalid column name. Available columns:", ", ".join(available_columns))
                    continue
                
                # Show top 10 values for the selected column
                df = pd.read_csv(csv_file)
                top_values = get_top_values(df, column)
                print(f"\nTop 5 most frequent values for {column}:")
                for value, count in top_values.items():
                    print(f"{value}: {count} occurrences")
                
                value = input(f"\nEnter value for {column}: ")
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