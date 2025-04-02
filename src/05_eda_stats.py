import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import scipy.stats as stats
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import shapiro, ttest_1samp, mannwhitneyu, wilcoxon, linregress
from collections import Counter

# Requirements:


nltk.download("vader_lexicon")
nltk.download("punkt")


def basic_summary(df):
    print("Total rows:", len(df))
    print("Unique conversations:", df['id'].nunique())
    print("From counts:")
    print(df['from'].value_counts())
    print("\nColumn names:", df.columns.tolist())

    messages_per_chat = df.groupby('id').size()
    print("\nMessages per chat:")
    print("Mean:", round(messages_per_chat.mean(), 4))
    print("Std:", round(messages_per_chat.std(), 4))
    print("Quantiles:")
    print(messages_per_chat.quantile([0.25, 0.5, 0.75]))

    # Additional stats from code_0320
    grouped = df.groupby('id')['value'].count()
    print("\nAdditional Message Stats:")
    print("25%:", grouped.quantile(0.25))
    print("50%:", grouped.quantile(0.5))
    print("75%:", grouped.quantile(0.75))


def run_stat_tests(df, feature):
    print(f"\nRunning tests for {feature}:")
    values = df[feature].dropna()

    stat, p = shapiro(values)
    print(f"Shapiro-Wilk p-value: {p:.4f}")

    t_stat, t_p = ttest_1samp(values, 0.5)
    print(f"One-sample t-test vs 0.5: p-value = {t_p:.4f}")

    w_stat, w_p = wilcoxon(values - 0.5)
    print(f"Wilcoxon test vs 0.5: p-value = {w_p:.4f}")

    # Add Mann-Whitney for robustness
    m_stat, m_p = mannwhitneyu(values, [0.5]*len(values))
    print(f"Mann-Whitney U test vs 0.5: p-value = {m_p:.4f}")


def plot_wordcloud(df, col='value'):
    text = " ".join(df[col].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"WordCloud for {col}")
    plt.tight_layout()
    plt.show()


def sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['value'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df


def plot_feature_distribution(df, feature):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[feature].dropna(), bins=30, kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, features):
    corr = df[features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_path = "data/processed/alignment_features.json"
    df = pd.read_json(input_path)

    basic_summary(df)

    for f in ["care", "fairness", "loyalty", "authority", "sanctity"]:
        run_stat_tests(df, f)
        plot_feature_distribution(df, f)

    df = sentiment_analysis(df)
    plot_feature_distribution(df, "sentiment")
    plot_wordcloud(df)

    plot_correlation_matrix(df, ["care", "fairness", "loyalty", "authority", "sanctity", "sentiment"])