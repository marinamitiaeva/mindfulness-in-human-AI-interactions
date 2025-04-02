# Install required packages if not already installed
install.packages(c("arrow", "dplyr", "politeness", "ggplot2", "purrr", "jsonlite", "stringr", "tm", "textstem", "spacyr", "jsonlite"))

# Load necessary libraries
library(arrow)
library(dplyr)
library(politeness)
library(ggplot2)
library(purrr)
library(stringr)
library(tm)
library(textstem)
library(jsonlite)
library(spacyr)
spacy_download_langmodel()
library(spacyr)
spacy_initialize(model = "en_core_web_sm")

# Check if the function exists
print(exists("str_squish", where = "package:stringr"))

# Preprocessing function for the 'value' column
preprocess_for_politeness <- function(df, text_column = "value") {
  if (!text_column %in% colnames(df)) {
    stop(paste("Column", text_column, "not found in the dataframe"))
  }
  
  basic_text <- tolower(df[[text_column]])  # Convert to lowercase
  basic_text <- str_squish(basic_text)  # Remove extra whitespace
  basic_text <- str_replace_all(basic_text, 
                                "(https?://\\S+|www\\.\\S+|\\S+@\\S+)", "")  # Remove URLs/emails
  basic_text <- str_replace_all(basic_text, "[^a-zA-Z0-9\\s]", "")  # Remove special characters
  
  df$basic_clean_text <- basic_text
  return(df)
}

postprocess_tm_lemmatize <- function(df, text_column = "basic_clean_text") {
  if (!text_column %in% colnames(df)) {
    stop(paste("Column", text_column, "not found in the dataframe"))
  }
  
  # Remove stop words and lemmatize
  clean_text <- sapply(df[[text_column]], function(text) {
    words <- unlist(strsplit(text, "\\s+"))
    words <- words[!words %in% stopwords("en")]  # Remove stop words
    words <- lemmatize_words(words)  # Lemmatize
    paste(words, collapse = " ")
  })
  
  df$fully_clean_text <- clean_text
  return(df)
}

# Read the CSV file
df1 <- read.csv("~/Downloads/sg_90k_part1_clean_en_unflattened.csv", header = TRUE)
df1 <- df1 %>% select(-X)
df2 <- read.csv("~/Downloads/sg_90k_part2_clean_en_unflattened.csv", header = TRUE)

head(df1)

df <- rbind(df1, df2)

# Apply basic cleaning
df <- preprocess_for_politeness(df, text_column = "value")

# Apply stopword removal and lemmatization
df <- postprocess_tm_lemmatize(df, text_column = "basic_clean_text")

# Check the preprocessed data
head(df)

# Split the dataframe into chunks for processing
chunk_size <- ceiling(nrow(df) / 20)
df_list <- split(df, rep(1:20, each = chunk_size, length.out = nrow(df)))

# Initialize a list to store politeness results
df_pol_list <- list()

# Apply politeness function to each chunk
for (i in seq_along(df_list)) {
  tryCatch({
    df_pol_list[[i]] <- politeness(df_list[[i]]$basic_clean_text)
    cat("Politeness applied to chunk", i, "\n")
  }, error = function(e) {
    cat("Error in chunk", i, ":", conditionMessage(e), "\n")
  })
}

# Combine politeness results and original data
combined_df_pol <- bind_rows(df_pol_list)
final_df <- cbind(bind_rows(df_list), combined_df_pol)

# Function to count words in the 'value' column
count_words <- function(df) {
  df %>%
    mutate(
      word_count = sapply(basic_clean_text, function(x) length(unlist(strsplit(x, "\\s+")))),
      stopword_count = sapply(basic_clean_text, function(x) {
        words <- tolower(unlist(strsplit(x, "\\s+")))
        sum(words %in% tm::stopwords("en"))  # Use stopwords("en") from tm package
      })
    )
}

# Add word counts to the dataframe
final_df <- count_words(final_df)

# Function to normalize politeness scores
normalize_columns <- function(df) {
  cols_to_normalize <- setdiff(names(df), c("id", "value", "from", "word_count", "basic_clean_text", "fully_clean_text"))
  df %>%
    mutate(across(all_of(cols_to_normalize), ~ . / word_count, .names = "norm_{col}"))
}

# Normalize politeness scores
final_df_normalized <- normalize_columns(final_df)

add_internal_id <- function(df) {
  df$internal_id <- seq(0, nrow(df) - 1)
  return(df)
}

final_df_normalized <- add_internal_id(final_df_normalized) 

tail(final_df_normalized$internal_id)

write_json(final_df_normalized, path = "~/Downloads/final_df_normalized_full.json", pretty = TRUE)

selected_columns <- c("internal_id", "id", "from", "value", "basic_clean_text", "fully_clean_text", "word_count", "stopword_count", 
                      "Hedges", "Swearing", "Reassurance", "Please", "Gratitude", "Apology", "Affirmation", 
                      "norm_Hedges", "norm_Swearing", "norm_Reassurance", "norm_Please", "norm_Gratitude", 
                      "norm_Apology", "norm_Affirmation", "norm_stopword_count")

df_to_save <- final_df_normalized[, selected_columns]
write_json(df_to_save, path = "~/Downloads/final_df_normalized_shrt.json", pretty = TRUE)