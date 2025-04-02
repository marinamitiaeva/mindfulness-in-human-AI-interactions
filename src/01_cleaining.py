import os
import json
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent language detection
DetectorFactory.seed = 0  

def load_and_flatten_all_json(json_dir):
    """Load and flatten all ShareGPT JSON files in a folder."""
    all_rows = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for filename in json_files:
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]  # wrap in list if single object

        for item in data:
            conv_id = item.get("id")
            for turn in item.get("conversations", []):
                all_rows.append({
                    "id": conv_id,
                    "from": turn.get("from"),
                    "value": turn.get("value")
                })
    
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["value"])
    df["value"] = df["value"].str.strip()
    df = df[df["value"] != ""]  # remove empty strings
    return df

def detect_language_flags(df):
    """Detect language per conversation and assign flags."""
    lang_map = {}
    for conv_id, group in df.groupby("id"):
        langs = []
        for text in group["value"]:
            try:
                langs.append(detect(text))
            except LangDetectException:
                continue
        lang_set = set(langs)
        if not lang_set:
            lang_map[conv_id] = 2  # no English detected
        elif lang_set == {"en"}:
            lang_map[conv_id] = 0  # pure English
        else:
            lang_map[conv_id] = 1  # mixed
    df["mixed_language_flag"] = df["id"].map(lang_map)
    return df

def save_cleaned_data(df, output_path):
    """Save processed dataframe to JSON."""
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)

# ========== RUN PIPELINE ==========

if __name__ == "__main__":
    input_dir = "data/raw"
    output_file = "data/cleaned/cleaned_conversations.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("Loading and flattening data...")
    df = load_and_flatten_all_json(input_dir)

    print("Detecting languages and assigning flags...")
    df = detect_language_flags(df)

    print(f"Saving cleaned data to {output_file}...")
    save_cleaned_data(df, output_file)

    print("Done! Total rows:", len(df))