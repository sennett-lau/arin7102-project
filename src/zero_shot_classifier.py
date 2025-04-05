import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score
import random
from tqdm import tqdm
import torch

def run_zero_shot_classification():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('filtered_combined_medical_tldr_dataset.csv')
    
    # Take first 5000 samples
    df = df.head(5000)
    print(f"Processing {len(df)} samples...")
    
    # Calculate text lengths
    df['text_length'] = df['text'].str.len()
    
    # Initialize zero-shot classifier
    print("Initializing zero-shot classifier...")
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=0 if torch.cuda.is_available() else -1)
    
    # Define candidate labels
    candidate_labels = [
        "drug review",
        "general text"
    ]
    
    # Function to classify a single text
    def classify_text(text):
        result = classifier(text, candidate_labels)
        # Return 1 if "drug review" has highest score, 0 otherwise
        return 1 if result['labels'][0] == "drug review" else 0
    
    # Take a sample for detailed analysis
    sample_size = 5
    sample_df = df.groupby('label').apply(lambda x: x.sample(sample_size)).reset_index(drop=True)
    
    print("\nAnalyzing sample texts...")
    for _, row in sample_df.iterrows():
        text = row['text'][:500]  # Truncate text for display
        actual_label = row['label']
        result = classifier(text, candidate_labels)
        
        print(f"\nText: {text[:200]}...")
        print(f"Text length: {len(text)} characters")
        print(f"Actual label: {'Drug Review' if actual_label == 1 else 'Not Drug Review'}")
        print("Predictions:")
        for label, score in zip(result['labels'], result['scores']):
            print(f"- {label}: {score:.4f}")
        print("-" * 80)
    
    # Process all texts with a progress bar
    print("\nProcessing dataset...")
    predictions = []
    for text in tqdm(df['text'], desc="Classifying texts"):
        pred = classify_text(text)
        predictions.append(pred)
    
    # Calculate metrics
    accuracy = accuracy_score(df['label'], predictions)
    report = classification_report(df['label'], predictions)
    
    # Add predictions to dataframe
    df['zero_shot_prediction'] = predictions
    
    # Analyze text length impact
    print("\nText Length Analysis:")
    print("\nAverage text length by actual label:")
    print(df.groupby('label')['text_length'].mean())
    
    print("\nAverage text length by prediction:")
    print(df.groupby('zero_shot_prediction')['text_length'].mean())
    
    # Calculate accuracy by text length quartiles
    df['length_quartile'] = pd.qcut(df['text_length'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    accuracy_by_length = df.groupby('length_quartile').apply(
        lambda x: accuracy_score(x['label'], x['zero_shot_prediction'])
    )
    print("\nAccuracy by text length quartile:")
    print(accuracy_by_length)
    
    print("\nClassification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    
    # Save predictions
    df.to_csv('zero_shot_predictions_5000.csv', index=False)
    print("\nPredictions saved to 'zero_shot_predictions_5000.csv'")
    
    # Show some examples of correct and incorrect predictions
    print("\nExamples of Correct Predictions:")
    correct_preds = df[df['label'] == df['zero_shot_prediction']].sample(3)
    for _, row in correct_preds.iterrows():
        print(f"\nText: {row['text'][:200]}...")
        print(f"Text length: {row['text_length']} characters")
        print(f"Actual and Predicted Label: {'Drug Review' if row['label'] == 1 else 'Not Drug Review'}")
        print("-" * 80)
    
    print("\nExamples of Incorrect Predictions:")
    incorrect_preds = df[df['label'] != df['zero_shot_prediction']].sample(3)
    for _, row in incorrect_preds.iterrows():
        print(f"\nText: {row['text'][:200]}...")
        print(f"Text length: {row['text_length']} characters")
        print(f"Actual Label: {'Drug Review' if row['label'] == 1 else 'Not Drug Review'}")
        print(f"Predicted Label: {'Drug Review' if row['zero_shot_prediction'] == 1 else 'Not Drug Review'}")
        print("-" * 80)

if __name__ == "__main__":
    run_zero_shot_classification() 