# alignment_processor.py

import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def lexical_overlap(text1, text2):
    tokens1 = set(word_tokenize(text1.lower()))
    tokens2 = set(word_tokenize(text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def syntactic_similarity(text1, text2):
    vec = CountVectorizer().fit([text1, text2])
    vecs = vec.transform([text1, text2])
    return cosine_similarity(vecs)[0, 1]

def compute_alignment_features(df):
    overlaps = []
    similarities = []
    
    grouped = df.groupby("id")
    
    for _, group in tqdm(grouped):
        texts = group.sort_values("from")["value"].tolist()
        if len(texts) < 2:
            overlaps.append(None)
            similarities.append(None)
            continue
        overlap = lexical_overlap(texts[0], texts[1])
        similarity = syntactic_similarity(texts[0], texts[1])
        overlaps.append(overlap)
        similarities.append(similarity)

    df["lexical_overlap"] = overlaps
    df["syntactic_similarity"] = similarities
    return df

if __name__ == "__main__":
    input_path = "data/processed/morality_scores.json"
    output_path = "data/processed/alignment_features.json"

    df = pd.read_json(input_path)
    df = compute_alignment_features(df)
    df.to_json(output_path, orient="records", indent=2, force_ascii=False)
    print(f"Saved alignment features to {output_path}")
