import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Add time module for measuring execution time
from utils.load_data import load_webmd_drug_reviews_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
tf.random.set_seed(42)

import torch
from sentence_transformers import SentenceTransformer
import gensim.downloader as downloader

# Create results directory if it doesn't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, 'results', 'review-pred-innovation')
os.makedirs(results_dir, exist_ok=True)

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower()
    return text

def load_embedding_model():
    embeddings = downloader.load("glove-wiki-gigaword-200")
    all_words = list(embeddings.index_to_key)
    print(f"Loaded vocab size {len(all_words)}")
    return {word: embedding for word, embedding in zip(all_words, embeddings)}

def train_model(model, model_name, target_name, X_train, y_train):
    if model_name == "LSTM":
        y_train = to_categorical(y_train - 1, num_classes=5)

    # Measure training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    with open(f'{results_dir}/{model_name}_{target_name}.pkl','wb') as f:
        pickle.dump(model, f)
    
    return train_time

def evaluate_model(model_name, target_name, X_val, y_val):
    with open(f'{results_dir}/{model_name}_{target_name}.pkl','rb') as f:
        model = pickle.load(f)

    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_val)
    predict_time = time.time() - start_time

    if model_name == "LSTM": y_pred = np.argmax(y_pred, axis=-1) + 1
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    residuals = y_val - y_pred
    acc = 1 - np.count_nonzero(residuals) / len(residuals)
    
    return {
        'metrics': {
            'acc': acc,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
        },
        'PredictTime': predict_time,
        'residuals': residuals,
        'predictions': y_pred,
        'actual': y_val
    }

def evaluate_innovation(target_name, X_test, y_test):
    y_preds = np.zeros((len(y_test), 4))
    y_pred_final = np.zeros(len(y_test))
    # Measure prediction time
    start_time = time.time()
    for i, model_name in enumerate(["LR", "MLP", "RF", "LSTM"]):
        with open(f'{results_dir}/{model_name}_{target_name}.pkl','rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        if model_name == "LSTM": y_pred = np.argmax(y_pred, axis=-1) + 1
        y_preds[:, i] = y_pred
    for i, row in enumerate(y_preds):
        y_pred_final[i] = row[np.argmax(np.bincount(row))] if np.max(np.bincount(row)) > 1 else round(np.mean(row))

    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred_final)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_final)
    mae = mean_absolute_error(y_test, y_pred_final)
    residuals = y_test - y_pred_final
    acc = 1 - np.count_nonzero(residuals) / len(residuals)
    predict_time = time.time() - start_time

    return {
        'metrics': {
            'acc': acc,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
        },
        'PredictTime': predict_time,
        'residuals': residuals,
        'predictions': y_pred_final,
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
    plt.title(f'Model: {model_name} | Target: {target_name}\nPredicted vs. Actual Values', fontsize=12)
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
    plt.title(f'Model: {model_name} | Target: {target_name}\nResiduals vs. Predicted Values', fontsize=12)
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
    plt.title(f'Model: {model_name} | Target: {target_name}\nDistribution of Residuals', fontsize=12)
    plt.grid(True)
    plt.savefig(f'{results_dir}/{model_name}_{target_name}_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv(results):
    # Save metrics summary
    metrics_data = []
    for target_name, target_results in results.items():
        for model_name, result in target_results.items():
            metrics_data.append({
                'Model': model_name,
                'Target': target_name,
                'acc': result['metrics']['acc'],
                'MSE': result['metrics']['MSE'],
                'RMSE': result['metrics']['RMSE'],
                'R2': result['metrics']['R2'],
                'MAE': result['metrics']['MAE'],
                'TrainTime': result['TrainTime'],  # Add training time
                'PredictTime': result['PredictTime']  # Add prediction time
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f'{results_dir}/model_metrics_summary.csv', index=False)
    
    # Save predictions and actual values for each model and target
    for target_name, target_results in results.items():
        for model_name, result in target_results.items():
            predictions_df = pd.DataFrame({
                'Actual': result['actual'],
                'Predicted': result['predictions'],
                'Residuals': result['residuals']
            })
            predictions_df.to_csv(f'{results_dir}/{model_name}_{target_name}_predictions.csv', index=False)

def main():
    # Load data
    print("Loading data...")
    df = load_webmd_drug_reviews_dataset()
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_reviews'] = df['Reviews'].apply(preprocess_text)

    # Create sentence embeddings
    print("Creating sentence embeddings...")
    embed_model_full_name = "sentence-transformers/all-mpnet-base-v2" # sentence-transformers/all-MiniLM-L6-v2
    embed_model_name = embed_model_full_name[embed_model_full_name.find('/') + 1:]
    embed_file_path = os.path.join(results_dir, f'{embed_model_name}.pt')

    if os.path.exists(embed_file_path):
        X_embed = torch.load(embed_file_path, weights_only=False)
    else:
        embed_model = SentenceTransformer(embed_model_full_name)
        X_embed = embed_model.encode(df['processed_reviews'])
        torch.save(X_embed, embed_file_path)
    
    print(f"X_embed shape: {X_embed.shape}")
    X_embed = X_embed

    """ glove embeddings, unused
    dictionary = load_embedding_model()
    embed_list = []
    for review in df['processed_reviews']:
        embed, i = np.zeros(200), 0
        for word in review.split():
            if word in dictionary:
                embed += dictionary[word]
                i += 1
        embed_list.append(embed / max(1, i))
    X_embed = np.array(embed_list)
    """
    
    # Define targets and models
    target_names = ['EaseofUse', 'Effectiveness', 'Satisfaction']

    model_lstm_ease = Sequential([
        LSTM(256, input_shape=(X_embed.shape[1],1)),
        Dense(5, activation='softmax')
    ])
    model_lstm_ease.compile(optimizer='adam', loss='categorical_crossentropy')
    model_lstm_effect = Sequential([
        LSTM(256, input_shape=(X_embed.shape[1],1)),
        Dense(5, activation='softmax')
    ])
    model_lstm_effect.compile(optimizer='adam', loss='categorical_crossentropy')
    model_lstm_sat = Sequential([
        LSTM(256, input_shape=(X_embed.shape[1],1)),
        Dense(5, activation='softmax')
    ])
    model_lstm_sat.compile(optimizer='adam', loss='categorical_crossentropy')

    models = {
        "EaseofUse": {
            "LR": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
            "MLP": MLPClassifier(max_iter=1000, batch_size=1024, random_state=42),
            "RF": RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42),
            "LSTM": model_lstm_ease,
        },
        "Effectiveness": {
            "LR": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
            "MLP": MLPClassifier(max_iter=1000, batch_size=1024, random_state=42),
            "RF": RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42),
            "LSTM": model_lstm_effect,
        },
        "Satisfaction": {
            "LR": LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42),
            "MLP": MLPClassifier(max_iter=1000, batch_size=1024, random_state=42),
            "RF": RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=42),
            "LSTM": model_lstm_sat,
        }
    }

    train_val_test_splits = {}
    for target in target_names:
        X_train, X_val, y_train, y_val = train_test_split(X_embed, df[target], test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        train_val_test_splits[target] = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test
        }

    # Train and evaluate models
    results = {}
    
    # print('Model\t\t\tTarget\t\t\tMSE\t\t\tR2\t\t\tRMSE\t\t\tMAE\t\t\tTrainTime(s)\t\t\tPredictTime(s)')
    
    for target_name, target_models in models.items():
        results[target_name] = {}
        for model_name, model in target_models.items():
            print(f"Training {model_name} for {target_name}...")
            # Train and evaluate models
            #if model_name in ["RF", "LSTM"]:
            #    train_time = train_model(model, model_name, target_name, train_val_test_splits[target_name]["X_train"][:50000], train_val_test_splits[target_name]["y_train"][:50000])
            #else:
            #    train_time = train_model(model, model_name, target_name, train_val_test_splits[target_name]["X_train"], train_val_test_splits[target_name]["y_train"])
            print(f"Evaluating {model_name} for {target_name}...")
            r = evaluate_model(model_name, target_name, train_val_test_splits[target_name]["X_val"], train_val_test_splits[target_name]["y_val"])
            r["TrainTime"] = 0#train_time
            # print(f"{model_name}\t{target}\t\t{r['metrics']['MSE']:.4f}\t\t\t{r['metrics']['R2']:.4f}\t\t\t{r['metrics']['RMSE']:.4f}\t\t\t{r['metrics']['MAE']:.4f}\t\t\t{r['metrics']['TrainTime']:.4f}\t\t\t{r['metrics']['PredictTime']:.4f}")
            
            results[target_name][model_name] = r
            
            # Create and save plots
            plot_predictions_vs_actual(r['actual'], r['predictions'], target_name, model_name)
            plot_residuals(r['residuals'], r['predictions'], target_name, model_name)

    # innovation
    for target_name, target_models in models.items():
        print(f"Evaluating INNO for {target_name}...")
        r = evaluate_innovation(target_name, train_val_test_splits[target_name]["X_test"], train_val_test_splits[target_name]["y_test"])
        r["TrainTime"] = 0
        results[target_name]["INNO"] = r
    
    # Save results to CSV
    print("Saving results to CSV...")
    save_results_to_csv(results)
    
    print(f"All results saved to {results_dir}")

if __name__ == "__main__":
    main()