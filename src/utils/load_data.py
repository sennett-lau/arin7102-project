import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")

def load_webmd_drug_reviews_dataset():
    path = os.path.join(data_dir, "webmd.csv")
    df = pd.read_csv(path)
    
    return df
