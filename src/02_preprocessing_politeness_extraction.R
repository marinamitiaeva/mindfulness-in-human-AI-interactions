# =============================================================================
# 02_politeness_preprocessing.R
# Description: Preprocess ShareGPT conversation text for politeness analysis
# =============================================================================

# -----------------------------
# Install and Load Required Packages
# -----------------------------
packages <- c("arrow", "dplyr", "politeness", "ggplot2", "purrr", "jsonlite",
              "stringr", "tm", "textstem", "spacyr")

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)
if (length(to_install)) install.packages(to_install)

lapply(packages, library, character.only = TRUE)

# Initialize spaCy
spacy_download_langmodel()
spacy_initialize(model = "en_core_web_sm")

# -----------------------------
# Preprocessing Functions
# -----------------------------

# Clean and normalize the input column
preprocess_for_politeness <- function(df, text_column = "value") {
  stopifnot(text_column %in% colnames(df))
  
  basic_text <- tolower(df[[text_column]])
  basic_text <- str_squish(basic_text)
  basic_text <- str_replace_all(basic_text, "(https?://\\S+|www\\.\\S+|\\S+@\\S+)", "")
  basic_text <- str_replace_all(basic_text, "[^a-zA-Z0-9\\s]", "")
  
  df$basic_clean_text <- basic_text
  return(df)
}

# Remove stopwords and lemmatize
postprocess_tm_lemmatize <- function(df, text_column = "basic_clean_text") {
  stopifnot(text_column %in% colnames(df))
  
  clean_text <- sapply(df[[text_column]], function(text) {
    words <- unlist(strsplit(text, "\\s+"))
    words <- words[!words %in% stopwords("en")]
    words <- lemmatize_words(words)
    paste(words, collapse = " ")
  })
  
  df$fully_clean_text <- clean_text
  return(df)
}

# Count words and stopwords
count_words <- function(df) {
  df %>%
    mutate(
      word_count = sapply(basic_clean_text, function(x) length(unlist(strsplit(x, "\\s+")))),
      stopword_count = sapply(basic_clean_text, function(x) {
        words <- tolower(unlist(strsplit(x, "\\s+")))
        sum(words %in% stopwords("en"))
      })
    )
}

# Normalize politeness columns
normalize_columns <- function(df) {
  cols_to_normalize <- setdiff(names(df), c("id", "value", "from", "word_count", 
                                            "basic_clean_text", "fully_clean_text"))
  df %>%
    mutate(across(all_of(cols_to_normalize), ~ . / word_count, .names = "norm_{col}"))
}

# Add unique internal ID
add_internal_id <- function(df) {
  df$internal_id <- seq_len(nrow(df)) - 1
  return(df)
}

# -----------------------------
# Load and Combine Data
# -----------------------------

df1 <- read.csv("~/Downloads/sg_90k_part1_clean_en_unflattened.csv")
df2 <- read.csv("~/Downloads/sg_90k_part2_clean_en_unflattened.csv")
df1 <- df1 %>% select(-X)
df <- bind_rows(df1, df2)

# -----------------------------
# Apply Preprocessing
# -----------------------------

df <- preprocess_for_politeness(df)
df <- postprocess_tm_lemmatize(df)
df <- count_words(df)

# -----------------------------
# Apply Politeness Function in Chunks
# -----------------------------

chunk_size <- ceiling(nrow(df) / 20)
df_list <- split(df, rep(1:20, each = chunk_size, length.out = nrow(df)))

df_pol_list <- list()
for (i in seq_along(df_list)) {
  tryCatch({
    df_pol_list[[i]] <- politeness(df_list[[i]]$basic_clean_text)
    cat("✔ Politeness processed for chunk", i, "\n")
  }, error = function(e) {
    cat("✘ Error in chunk", i, ":", conditionMessage(e), "\n")
  })
}

combined_df_pol <- bind_rows(df_pol_list)
final_df <- cbind(bind_rows(df_list), combined_df_pol)

# -----------------------------
# Final Transformations & Save
# -----------------------------

final_df <- normalize_columns(final_df)
final_df <- add_internal_id(final_df)

# Save full version
write_json(final_df, path = "~/Downloads/final_df_normalized_full.json", pretty = TRUE)

# Select relevant columns for export
selected_columns <- c("internal_id", "id", "from", "value", "basic_clean_text", 
                      "fully_clean_text", "word_count", "stopword_count", 
                      "Hedges", "Swearing", "Reassurance", "Please", "Gratitude", 
                      "Apology", "Affirmation", 
                      "norm_Hedges", "norm_Swearing", "norm_Reassurance", 
                      "norm_Please", "norm_Gratitude", "norm_Apology", 
                      "norm_Affirmation", "norm_stopword_count")

df_to_save <- final_df[, selected_columns]
# write_json(df_to_save, path = "~/Downloads/final_df_normalized_shrt.json", pretty = TRUE)