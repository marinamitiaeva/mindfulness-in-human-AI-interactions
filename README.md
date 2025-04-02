# Moral Foundations and Alignment Analysis

This repository contains a modular pipeline for analyzing language data through the lens of moral foundation theory and linguistic alignment. It includes three stages: moral score extraction, alignment computation, and statistical/visual analysis.

---

## 📁 Project Structure

```
data/
├── raw/                         # (Locally stored) Raw JSON files – not in repo
│
│   📦 Raw data files (~1.7GB total) are hosted externally:
│   👉 Download from Hugging Face: https://huggingface.co/datasets/RyokoAI/ShareGPT52K
├── processed/                  # Output from each pipeline stage
│   ├── morality_scores.json
│   ├── alignment_features.json
│   └── ...
scripts/
├── moralformer_processor.py      # Extracts moral foundation scores
├── alignment_processor.py        # Computes alignment metrics
├── moral_stats_and_eda.py        # Runs statistical tests and EDA
```

---

## ⚙️ Installation

Install all required Python libraries:

```bash
pip install pandas numpy torch transformers pyspark wordsegment matplotlib seaborn wordcloud nltk
```

Also download NLTK resources:

```python
import nltk
nltk.download("punkt")
nltk.download("vader_lexicon")
```

---

## 🚀 Usage

### Step 1: Moral Score Extraction

```bash
python scripts/moralformer_processor.py
```

- Uses `microsoft/moralformer-base` transformer
- Processes messages and outputs moral probabilities per message

### Step 2: Alignment Feature Computation

```bash
python scripts/alignment_processor.py
```

- Computes:
  - **Lexical overlap** (token intersection)
  - **Syntactic similarity** (cosine of vectorized messages)
- Saves output with added alignment features

### Step 3: Statistical Analysis and EDA

```bash
python scripts/moral_stats_and_eda.py
```

- Performs:
  - Descriptive stats
  - Shapiro-Wilk, t-tests, Wilcoxon, Mann-Whitney U tests
  - Sentiment scoring using VADER
  - Word clouds and feature distributions
  - Correlation heatmap across features

---

## 📊 Output

- `morality_scores.json`: Contains moral foundation probabilities per message
- `alignment_features.json`: Adds lexical and syntactic alignment scores
- Visuals (histograms, heatmaps, word clouds) are shown inline (or can be modified to save)

---

## 🧠 Moral Foundations

This project quantifies the presence of five moral dimensions in text using a transformer model:

- **Care** vs **Harm**
- **Fairness** vs **Cheating**
- **Loyalty** vs **Betrayal**
- **Authority** vs **Subversion**
- **Sanctity** vs **Degradation**

Scores range from 0 to 1 and are derived from Microsoft’s [`moralformer-base`](https://huggingface.co/microsoft/moralformer-base) model.

---

## 🙌 Acknowledgments

- [Moral Foundations Theory](https://moralfoundations.org/) — Graham et al.
- [Microsoft Research](https://huggingface.co/microsoft/moralformer-base)
- HuggingFace Transformers, NLTK, PySpark, Seaborn

---

## 📬 Contact

For questions, collaborations, or suggestions, please feel free to open an issue or get in touch via email.