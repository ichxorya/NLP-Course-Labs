#! Import libraries.
import os  
import pandas as pd 
import nltk 
import string  
import re 
import random 
import time  

# Download nltk packages for tokenization, stopwords and lemmatization.
nltk.download("punkt")  
nltk.download("stopwords") 
nltk.download("wordnet") 

# Import tokenizer, stopwords and lemmatizer.
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords 

stop_words = stopwords.words("english")

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


#! Function: Remove HTML tags from a string.
def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        text (str): string to remove HTML tags from.

    Return type:
        str: string without HTML tags.
    """
    return re.sub(r"<.*?>", "", text)


#! Function: Remove punctuation from a string.
def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from a string.

    Args:
        text (str): string to remove punctuation from.

    Return type:
        str: string without punctuation.
    """
    return text.translate(str.maketrans("", "", string.punctuation))


#! Function: Remove stopwords from a string.
def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from a string.

    Args:
        text (str): string to remove stopwords from.

    Return type:
        str: string without stopwords.
    """
    return " ".join([word for word in text.split() if word not in stop_words])


#! Function: Lemmatize a string.
def lemmatize(text: str) -> str:
    """
    Lemmatize a string (convert words to their base form)

    Args:
        text (str): string to lemmatize.

    Return type:
        str: lemmatized string.
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


#! Function: Clean a string.
def clean_text(text: str) -> str:
    """
    Clean a string.

    Args:
        text (str): string to clean.

    Return type:
        str: cleaned string.
    """
    text = remove_html_tags(text)
    text = text.lower() 
    text = remove_punctuation(text)  
    text = remove_stopwords(text)  
    text = lemmatize(text) 
    return text


#! Function: Clean a dataframe.
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a dataframe.
    The original dataframe is not modified.

    Args:
        data (pd.DataFrame): dataframe to clean.

    Return type:
        pd.DataFrame: cleaned dataframe.
    """
    # Clean the review column.
    cleaned_data = data.copy()
    cleaned_data["review"] = cleaned_data["review"].apply(clean_text)
    return cleaned_data


#! Function: Compare a random review before and after cleaning.
def compare_review_before_after_cleaning(
    dirty_data: pd.DataFrame, clean_data: pd.DataFrame, write_to_file: bool = False
) -> None:
    """
    Compare a random review before and after cleaning from two dataframes.
    The reviews are limited to 200 characters for better readability.
    The full comparison is also written to a file called
    'data_before_after_<time>.txt' (optional).

    Args:
        dirty_data (pd.DataFrame): dataframe containing the dirty data.
        clean_data (pd.DataFrame): dataframe containing the clean data.
    """

    # Get a random index.
    random_index = random.randint(0, min(dirty_data.shape[0], clean_data.shape[0]) - 1)

    # Get the dirty and clean review samples.
    review_samples = (
        dirty_data["review"].iloc[random_index],
        clean_data["review"].iloc[random_index],
    )
    review_kind = dirty_data["sentiment"].iloc[
        random_index
    ]  # Get the review kind (pos/neg).

    # Adjust the reviews for better readability.
    ## Add newlines after each 70 characters.
    review_samples_adjusted = [
        re.sub(r"(.{70})", r"\1\n", review) for review in review_samples
    ]
    ## Max width of each review is 200 characters.
    review_samples_adjusted = [
        review[:200] + "... (tldr)" if len(review) > 200 else review
        for review in review_samples_adjusted
    ]
    ## Add a tab before each line.
    review_samples_adjusted = [
        re.sub(r"^\s*", "\t", review, flags=re.MULTILINE)
        for review in review_samples_adjusted
    ]

    # Print the dirty and clean reviews.
    print(f"Comparing a random review ({review_kind}) before and after cleaning:")

    print("Dirty review:")
    print(review_samples_adjusted[0])
    print()

    print("Clean review:")
    print(review_samples_adjusted[1])
    print()

    # Write the dirty and clean reviews to a file called 'data_before_after_<time>.txt'.
    if write_to_file:
        current_time = time.strftime("%Y%m%d-%H%M%S")

        with open(f"data_before_after_{current_time}.txt", "w") as f:
            f.write(
                f"Comparing a random review ({review_kind}) before and after cleaning:\n\n"
            )
            f.write("Dirty review:\n")
            f.write("\t" + review_samples[0] + "\n\n")
            f.write("Clean review:\n")
            f.write("\t" + review_samples[1] + "\n\n")

    # Print a line to separate the output.
    print("-" * 100)


#! Function: Save the cleaned data to a csv file and text files.
def save_cleaned_data(data: pd.DataFrame, is_train_data: bool) -> None:
    """
    Save the cleaned data to a csv file.

    Args:
        data (pd.DataFrame): dataframe to save.
    """
    csv_file_name = "train.csv" if is_train_data else "test.csv"

    # Save the dataframe to a csv file.
    data.to_csv(f"cleaned_aclImdb/{csv_file_name}", index=False)

    print(f"Saved the cleaned data to 'cleaned_aclImdb/csv/{csv_file_name}'")

    # Print a line to separate the output.
    print("-" * 100)
