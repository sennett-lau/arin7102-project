import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from tqdm import tqdm
import gc
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize models
extractive_model = SentenceTransformer('all-MiniLM-L6-v2')
model_name = "allenai/led-base-16384-ms2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

def extractive_summary(review, query="Summarize the key points of this review", top_k=1):
    """
    Generate an extractive summary for a single review using sentence embeddings.
    """
    try:
        sentences = sent_tokenize(review)
        if not sentences:
            return ""
        
        # Encode sentences and query
        sentence_embeddings = extractive_model.encode(sentences, convert_to_tensor=True)
        query_embedding = extractive_model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
        
        # Select top-k sentences
        top_k_indices = np.argsort(cos_scores.cpu().numpy())[-top_k:]
        top_k_sentences = [sentences[i] for i in sorted(top_k_indices)]
        
        return " ".join(top_k_sentences)
    except Exception as e:
        print(f"Error in extractive summary: {str(e)}")
        return ""

def estimate_tokens(text):
    """
    Estimate the number of tokens in a text using the LED tokenizer, including special tokens.
    """
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    except Exception as e:
        print(f"Error estimating tokens: {str(e)}")
        return 0

def classify_summary(summary):
    """
    Classify an extractive summary into Effectiveness, Side Effects, or Overall Experience.
    """
    labels = ["Effectiveness", "Side Effects", "Overall Experience"]
    try:
        result = classifier(summary, candidate_labels=labels, multi_label=False)
        return result['labels'][0]
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return "Overall Experience"  # Default to Overall if classification fails

def deduplicate_summaries(summaries):
    """
    Remove near-identical summaries based on cosine similarity.
    """
    if not summaries:
        return []
    
    embeddings = extractive_model.encode(summaries, convert_to_tensor=True)
    cos_scores = util.cos_sim(embeddings, embeddings)
    threshold = 0.9  # Similarity threshold for deduplication
    
    unique_indices = []
    seen = set()
    for i in range(len(summaries)):
        if i not in seen:
            unique_indices.append(i)
            for j in range(i + 1, len(summaries)):
                if cos_scores[i][j] > threshold:
                    seen.add(j)
    
    return [summaries[i] for i in unique_indices]

def polish_summary(summary):
    """
    Polish the summary to reduce repetitive 'Users' and improve natural phrasing.
    """
    # Reduce repetitive 'Users'
    sentences = sent_tokenize(summary)
    polished_sentences = []
    user_count = 0
    for sentence in sentences:
        if 'Users' in sentence:
            user_count += 1
            if user_count > 1:
                sentence = sentence.replace('Users', 'They', 1)
        polished_sentences.append(sentence)
    
    summary = " ".join(polished_sentences)
    
    # Additional cleanup
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = summary[0].upper() + summary[1:]  # Capitalize first letter
    
    return summary

def summarize_category(summaries, category, max_length, min_length, max_input_tokens=16384):
    """
    Generate an abstractive summary for a list of summaries in a specific category.
    """
    if not summaries:
        return ""

    # Batch summaries to stay within token limit
    batches = []
    current_batch = []
    current_tokens = 0
    separator_tokens = estimate_tokens("</s>")

    for summary in summaries:
        summary_tokens = estimate_tokens(summary)
        if current_tokens + summary_tokens + separator_tokens <= max_input_tokens:
            current_batch.append(summary)
            current_tokens += summary_tokens + separator_tokens
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [summary]
            current_tokens = summary_tokens + separator_tokens

    if current_batch:
        batches.append(current_batch)

    # Abstractive summarization for each batch
    batch_summaries = []
    for batch in tqdm(batches, desc=f"Processing {category} batches"):
        concatenated_text = "</s>".join(batch)
        try:
            inputs = tokenizer(
                concatenated_text,
                max_length=max_input_tokens,
                truncation=True,
                return_tensors="pt"
            ).to(model.device)
            
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=0.5,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            batch_summaries.append(summary)
        except Exception as e:
            print(f"Error in {category} batch summarization: {str(e)}")
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    if not batch_summaries:
        return ""

    # Deduplicate batch summaries
    batch_summaries = deduplicate_summaries(batch_summaries)

    # Filter contradictory summaries for Overall Experience
    if category == "Overall Experience":
        sentiment_scores = [(s, TextBlob(s).sentiment.polarity) for s in batch_summaries]
        # Keep summaries with balanced sentiment (not too extreme)
        filtered_summaries = [s for s, score in sentiment_scores if -0.3 < score < 0.3]
        batch_summaries = filtered_summaries if filtered_summaries else batch_summaries[:1]  # Fallback to first summary

    # Summarize the batch summaries
    concatenated_batch_summaries = "</s>".join(batch_summaries)
    try:
        inputs = tokenizer(
            concatenated_batch_summaries,
            max_length=max_input_tokens,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)
        
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=0.5,
            early_stopping=True
        )
        final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in final {category} summarization: {str(e)}")
        final_summary = batch_summaries[0] if batch_summaries else ""  # Fallback to first summary
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    # Convert to third-person POV and polish
    final_summary = re.sub(r'\b(I|me)\b', 'Users', final_summary, flags=re.IGNORECASE)
    final_summary = re.sub(r'\bmy\b', 'their', final_summary, flags=re.IGNORECASE)
    final_summary = re.sub(r'\bI have\b', 'Users have', final_summary, flags=re.IGNORECASE)
    final_summary = re.sub(r'\bI am\b', 'Users are', final_summary, flags=re.IGNORECASE)
    final_summary = polish_summary(final_summary)

    return final_summary.strip()

def hierarchical_summarization(reviews, max_length=80, min_length=30, max_input_tokens=16384):
    """
    Perform hierarchical summarization with classification into categories.
    """
    if not reviews:
        return "No reviews provided."

    # Step 1: Extractive summarization for each review
    extractive_summaries = []
    for review in tqdm(reviews, desc="Generating extractive summaries"):
        summary = extractive_summary(str(review))
        if summary:
            extractive_summaries.append(summary)
    
    if not extractive_summaries:
        return "No valid extractive summaries generated."

    # Step 2: Classify extractive summaries
    categories = {
        "Effectiveness": [],
        "Side Effects": [],
        "Overall Experience": []
    }
    for summary in tqdm(extractive_summaries, desc="Classifying summaries"):
        category = classify_summary(summary)
        categories[category].append(summary)

    # Step 3: Summarize each category
    final_summary = f"### Summary of Cymbalta Reviews\n\n"
    for category in categories:
        summary_text = summarize_category(categories[category], category, max_length, min_length, max_input_tokens)
        if summary_text:
            final_summary += f"**{category}**\n{summary_text}\n\n"

    return final_summary.strip()

def load_and_filter_reviews(csv_path, drug_name):
    """
    Load reviews from a CSV file and filter by specified drug name.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'Drug' not in df.columns or 'Reviews' not in df.columns:
            raise ValueError("CSV must contain 'Drug' and 'Reviews' columns.")
        
        filtered_df = df[df['Drug'].str.lower() == drug_name.lower()]
        if filtered_df.empty:
            return None, f"No reviews found for drug: {drug_name}"
        
        reviews = filtered_df['Reviews'].dropna().tolist()
        return reviews, None
    except FileNotFoundError:
        return None, "CSV file not found."
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"

def get_top_drugs(csv_path, top_n=5):
    """
    Identify the top N drugs by review count in the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'Drug' not in df.columns:
            raise ValueError("CSV must contain 'Drug' column.")
        
        # Count reviews per drug (case-insensitive)
        drug_counts = df['Drug'].str.lower().value_counts()
        top_drugs = drug_counts.head(top_n).index.tolist()
        return top_drugs, None
    except FileNotFoundError:
        return None, "CSV file not found."
    except Exception as e:
        return None, f"Error processing CSV: {str(e)}"

# Example usage
if __name__ == "__main__":
    csv_path = "webmd.csv"
    
    # Get top 5 drugs
    top_drugs, error = get_top_drugs(csv_path, top_n=5)
    
    if error:
        print(error)
    else:
        output_file = "drug_summaries.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for drug_name in top_drugs:
                print(f"\nProcessing reviews for {drug_name}...")
                reviews, error = load_and_filter_reviews(csv_path, drug_name)
                
                if error:
                    summary = f"\nSummary of reviews for {drug_name}:\n{error}\n"
                    print(summary)
                    f.write(summary + "\n")
                else:
                    final_summary = hierarchical_summarization(reviews)
                    summary = f"\nSummary of reviews for {drug_name}:\n{final_summary}\n"
                    print(summary)
                    f.write(summary + "\n")
                    # Clear memory after each drug
                    torch.cuda.empty_cache()
                    gc.collect()
        
        print(f"\nSummaries saved to {output_file}")