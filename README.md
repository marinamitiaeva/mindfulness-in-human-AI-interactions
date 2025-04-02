# Mindful Interactions with GenAI: A ChatGPT Case Study

This repository accompanies our research paper **"Mindful Interactions with GenAI: A ChatGPT Case Study"**, which explores mindfulness in human–AI communcation through the lens of morality, politeness, and conversational alignment. The paper is currently under review in the *International Journal of Human-Computer Studies*.

---

## Goal

This project investigates whether generative AI systems like ChatGPT can engage in mindful communication—being present, polite, morally aware, and responsive to users.

We analyze 48,000 real Human–ChatGPT conversations across three core notions:
 - Morality
 - Politeness
 - Conversational Alignment
Our findings show that while ChatGPT consistently uses polite and moral language, it lacks adaptive flexibility—limiting its ability to respond to users with nuance. In contrast, human users show greater variation and emotional depth.

We propose a layered framework for mindful AI communication, and offer tools and insights for designing AI systems that are socially intelligent, ethically grounded, and capable of building trust in human–AI interactions.
Our aim is to offer a robust and reproducible pipeline for studying mindful and ethically grounded human–AI interactions.


## Data & Resources

- **Dataset: [ShareGPT52K (Hugging Face)](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)**  
  A large-scale collection of over 90,000 real human–ChatGPT conversations, used as the foundation for all turn-level and conversational analyses.

- **Morality Modeling: [Mformer (GitHub)](https://github.com/joshnguyen99/moral_axes)**  
  A transformer-based framework for scoring moral language across the five moral foundations defined by Moral Foundations Theory (MFT).

- **Politeness Detection: [`politeness` R package](https://cran.r-project.org/web/packages/politeness/vignettes/politeness.html)**  
  A linguistic tool for identifying politeness strategies in text, based on sociolinguistic theory and Brown & Levinson's Politeness Theory.

- **LIWC (Linguistic Inquiry and Word Count): [www.liwc.app](https://www.liwc.app)**  
  Used to compute Linguistic Style Matching (LSM) as part of Conversational Accommodation Theory (CAT).  
  *Note: LIWC requires a licensed dictionary.*

## Methods

Our analysis pipeline combines computational content analysis, linguistic signal processing, and statistical testing to examine how mindful communication manifests in human–ChatGPT interactions.

1. **Preprocessing**
   - Flattened the nested ShareGPT JSON into turn-level data
   - Filtered for English-only conversations
   - Cleaned and normalized text for consistent feature extraction

2. **Feature Extraction**
   - **Turn-level features**:
     - **Moral foundation probabilities** using the transformer-based Mformer model
     - **Politeness markers** based on Danescu et al.'s framework and the [`politeness`](https://cran.r-project.org/web/packages/politeness/vignettes/politeness.html) R package
     - **Sentiment scores** using VADER
     - **LIWC-style features** including pronouns, function words, and emotional tone
     - **Alignment metrics** between adjacent turns:
       - Lexical overlap  
       - Linguistic Style Matching (LSM)  
       - Politeness accommodation  
       - Sentiment alignment  

   - **Conversation-level summary features**:
     - Slopes of alignment metrics over the course of each conversation (to quantify adaptation)
     - Number of messages per conversation
     - Average message length
     - Vocabulary richness (vocab ratio)
     - Vocabulary ratio between GPT and human speakers

3. **Statistical Analysis**
   - **Normality testing** with Shapiro–Wilk
   - **Between-group comparisons** using Mann–Whitney U tests
   - **Within-group comparisons** using Wilcoxon signed-rank tests
   - **Spearman's Correlation analysis** to explore interdependencies across features
   - Comparative analysis between **human** and **GPT** speakers on all extracted metrics

4. **Visualization**
   - Word clouds to illustrate content patterns
   - Distribution plots for each linguistic/moral dimension
   - Slope and trend charts to visualize alignment behaviors over time

## 🗂 Repository Structure

### 📁 src/
Python and R scripts used as a pipeline for processing, analyzing, and extracting features from the data.

```
src/ 
├── 01_cleaning.py # Flatten and clean raw ShareGPT JSONs 
├── 02_preprocessing_politeness_extraction.R # Extract politeness markers using the politeness R package 
├── 03_morality_extraction.py # Apply MFormer to extract moral foundation scores 
├── 04_alignment_quantification.py # Compute alignment metrics (word overlap, LSM, sentiment, politeness) 
├── 05_eda_stats.py # Run statistical tests and exploratory data analysis 
└── ...
```

### 📁 notebooks/
Working Jupyter notebooks containing full analysis logic, exploratory experimentation, and intermediate results.

### 📁 results/
Contains generated results from statistical and visual analysis.

```
results/ 
├── figures/ # Visualizations (e.g., distributions, trends, word clouds) 
└── tests/ # Raw statistical test outputs (before z-scoring or Bonferroni correction)
```

### 📁 output/
```
output/ 
└── sample_metrics.csv # Cleaned text with extracted features for the first 100 rows (example output)
```

### 📄 requirements.txt
Python package dependencies for reproducing the analysis pipeline.
