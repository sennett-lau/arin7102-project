import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import gc
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize models
model_name = "allenai/led-base-16384-ms2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

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

def classify_review(review):
    """
    Classify a review into Effectiveness, Side Effects, or Overall Experience.
    """
    labels = ["Effectiveness", "Side Effects", "Overall Experience"]
    try:
        result = classifier(str(review), candidate_labels=labels, multi_label=False)
        return result['labels'][0]
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return "Overall Experience"  # Default to Overall if classification fails

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

def summarize_category(reviews, category, max_length=80, min_length=30, max_input_tokens=16384):
    """
    Generate an abstractive summary for a list of reviews in a specific category.
    """
    if not reviews:
        return ""

    # Batch reviews to stay within token limit
    batches = []
    current_batch = []
    current_tokens = 0
    separator_tokens = estimate_tokens("</s>")

    for review in reviews:
        review_tokens = estimate_tokens(str(review))
        if current_tokens + review_tokens + separator_tokens <= max_input_tokens:
            current_batch.append(str(review))
            current_tokens += review_tokens + separator_tokens
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [str(review)]
            current_tokens = review_tokens + separator_tokens

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
            summary = summary.strip()
            if summary:
                batch_summaries.append(summary)
        except Exception as e:
            print(f"Error in {category} batch summarization: {str(e)}")
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    if not batch_summaries:
        return ""

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
        final_summary = batch_summaries[0] if batch_summaries else ""
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

def summarize_reviews(reviews, max_length=80, min_length=30, max_input_tokens=16384):
    """
    Perform summarization with classification into categories, without extractive summarization.
    """
    if not reviews:
        return "No reviews provided."

    # Step 1: Classify reviews
    categories = {
        "Effectiveness": [],
        "Side Effects": [],
        "Overall Experience": []
    }
    for review in tqdm(reviews, desc="Classifying reviews"):
        category = classify_review(review)
        categories[category].append(str(review))

    # Step 2: Summarize each category
    final_summary = f"### Summary of Reviews\n\n"
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

def get_top_drugs(csv_path, top_n=1):
    """
    Identify the top N drugs by review count in the CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'Drug' not in df.columns:
            raise ValueError("CSV must contain 'Drug' column.")
        
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
        output_file = "drug_summaries_ablation_with_classification.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for drug_name in top_drugs:
                print(f"\nProcessing reviews for {drug_name}...")
                reviews, error = load_and_filter_reviews(csv_path, drug_name)
                
                if error:
                    summary = f"\nSummary of reviews for {drug_name}:\n{error}\n"
                    print(summary)
                    f.write(summary + "\n")
                else:
                    final_summary = summarize_reviews(reviews)
                    summary = f"\nSummary of reviews for {drug_name}:\n{final_summary}\n"
                    print(summary)
                    f.write(summary + "\n")
                    # Clear memory after each drug
                    torch.cuda.empty_cache()
                    gc.collect()
        
        print(f"\nSummaries saved to {output_file}")