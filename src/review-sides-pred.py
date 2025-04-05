import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Add time module for measuring execution time
from utils.load_data import load_webmd_drug_reviews_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# Create results directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results', 'review-sides-pred')
os.makedirs(results_dir, exist_ok=True)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error - more interpretable
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    
    residuals = y_test - y_pred
    
    return {
        'metrics': {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
            'TrainTime': train_time,  # Add training time
            'PredictTime': predict_time  # Add prediction time
        },
        'residuals': residuals,
        'predictions': y_pred,
        'actual': y_test
    }

def plot_predictions_vs_actual(y_test, y_pred, target_name, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate metrics for annotations
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Add metrics text box
    metrics_text = f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}"
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', fontsize=10)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Model: {model_name} | Target: {target_name}\nPredicted vs. Actual Values (Reviews + Side Effects)', fontsize=12)
    plt.grid(True)
    
    # Save figure
    plt.savefig(f'{results_dir}/{model_name}_{target_name}_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(residuals, y_pred, target_name, model_name):
    # Residuals vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Calculate stats for residuals
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    # Add residual stats text box
    stats_text = f"Mean: {mean_res:.4f}\nStd Dev: {std_res:.4f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', fontsize=10)
    
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Model: {model_name} | Target: {target_name}\nResiduals vs. Predicted Values (Reviews + Side Effects)', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{results_dir}/{model_name}_{target_name}_residuals_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    
    # Add residual stats text box to histogram too
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va='top', fontsize=10)
    
    plt.xlabel('Residuals')
    plt.title(f'Model: {model_name} | Target: {target_name}\nDistribution of Residuals (Reviews + Side Effects)', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{results_dir}/{model_name}_{target_name}_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv(results):
    # Save metrics summary
    metrics_data = []
    for model_name, model_results in results.items():
        for target, result in model_results.items():
            metrics_data.append({
                'Model': model_name,
                'Target': target,
                'MSE': result['metrics']['MSE'],
                'RMSE': result['metrics']['RMSE'],
                'R2': result['metrics']['R2'],
                'MAE': result['metrics']['MAE'],
                'TrainTime': result['metrics']['TrainTime'],  # Add training time
                'PredictTime': result['metrics']['PredictTime']  # Add prediction time
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f'{results_dir}/model_metrics_summary.csv', index=False)
    
    # Save predictions and actual values for each model and target
    for model_name, model_results in results.items():
        for target, result in model_results.items():
            predictions_df = pd.DataFrame({
                'Actual': result['actual'],
                'Predicted': result['predictions'],
                'Residuals': result['residuals']
            })
            predictions_df.to_csv(f'{results_dir}/{model_name}_{target}_predictions.csv', index=False)

def main():
    # Load data
    print("Loading data...")
    df = load_webmd_drug_reviews_dataset()
    
    # Handle missing values in both text fields
    df['Reviews'] = df['Reviews'].fillna('')
    df['Sides'] = df['Sides'].fillna('')
    
    # Preprocess both Reviews and Sides text
    print("Preprocessing text fields...")
    df['processed_reviews'] = df['Reviews'].apply(preprocess_text)
    df['processed_sides'] = df['Sides'].apply(preprocess_text)
    
    # Define targets and models
    targets = ['EaseofUse', 'Effectiveness', 'Satisfaction', 'UsefulCount']
    
    models = {
        'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
        'MLPClassification': MLPClassifier(max_iter=1000, batch_size=1024, random_state=42),
    }
    
    # Create TF-IDF features for both Reviews and Sides
    print("Creating TF-IDF features for Reviews and Side Effects...")
    tfidf_reviews = TfidfVectorizer(max_features=4000)
    tfidf_sides = TfidfVectorizer(max_features=1000)  # Fewer features for side effects as they're typically shorter
    
    # Transform both text fields
    X_reviews_tfidf = tfidf_reviews.fit_transform(df['processed_reviews'])
    X_sides_tfidf = tfidf_sides.fit_transform(df['processed_sides'])
    
    # Combine features by horizontally stacking the matrices
    X_combined = hstack([X_reviews_tfidf, X_sides_tfidf])
    
    # Train and evaluate models
    results = {}
    
    print('Model\t\t\tTarget\t\t\tMSE\t\t\tR2\t\t\tRMSE\t\t\tMAE\t\t\tTrainTime(s)\t\t\tPredictTime(s)')
    
    for model_name, model in models.items():
        results[model_name] = {}
        
        for target in targets:
            print(f"Training {model_name} for {target} using Reviews + Side Effects...")
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, df[target], test_size=0.2, random_state=42
            )
            
            # Train and evaluate models
            r = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            print(f"{model_name}\t{target}\t\t{r['metrics']['MSE']:.4f}\t\t\t{r['metrics']['R2']:.4f}\t\t\t{r['metrics']['RMSE']:.4f}\t\t\t{r['metrics']['MAE']:.4f}\t\t\t{r['metrics']['TrainTime']:.4f}\t\t\t{r['metrics']['PredictTime']:.4f}")
            
            results[model_name][target] = r
            
            # Create and save plots
            plot_predictions_vs_actual(r['actual'], r['predictions'], target, model_name)
            plot_residuals(r['residuals'], r['predictions'], target, model_name)
    
    # Save results to CSV
    print("Saving results to CSV...")
    save_results_to_csv(results)
    
    print(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main()