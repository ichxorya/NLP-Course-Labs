"""
This file contains utility functions for loading, visualizing and analyzing data in a dataframe.
"""

#! Import libraries.
import os
import pandas as pd 
import re 
import matplotlib.pyplot as plt
import seaborn as sns  
import time  


#! Function: Load data into a dataframe.
def load_data(dataset_path: str, dataset_type: str) -> pd.DataFrame:
    """
    Load data into a dataframe.

    Args:
        dataset_path (str): path to dataset.
        dataset_type (str): type of dataset (train, test).

    Return type:
        pd.DataFrame: dataframe containing the loaded data.

    Example:
    - Return a dataframe containing the training data.
                file        review        sentiment
            0   0_9.txt     good movie    pos
            1   1_2.txt     bad movie     neg
    """
    data = []  # list to store data

    # Check if dataset type is valid.
    if dataset_type not in ["train", "test"]:
        raise ValueError(
            f"""Invalid dataset type '{dataset_type}'. 
            Must be either 'train' or 'test'."""
        )

    # Iterate over labels.
    for label in ["pos", "neg"]:
        label_path = os.path.join(dataset_path, dataset_type, label)
        print(f"Loading data from {label_path}...")
        
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                data.append([file, f.read(), label])

    # Return the dataset as a dataframe.
    return pd.DataFrame(data, columns=["file", "review", "sentiment"])


#! Function: Visualize data from a dataframe.
def visualize_data(
    data: pd.DataFrame, title: str = "Example plot", write_to_file: bool = False
) -> None:
    """
    Note: The write_to_file functionality is currently broken.

    Visualize data from a dataframe as a countplot.
    Write the plot to a file called 'example_plot_<time>.png' (optional).

    Args:
        data (pd.DataFrame): dataframe containing the data.
        title (str): title of the plot.
    """
    
    # Plot the data.
    plt.figure(figsize=(10, 5))
    sns.countplot(x="sentiment", data=data)
    plt.title(title)

    # Save the plot to a file called 'example_plot_<time>.png'.
    if write_to_file:
        print("The write to file functionality is currently broken.")
    #     current_time = time.strftime("%Y%m%d-%H%M%S")
    #     plt.savefig(f"{title}_{current_time}.png")

    # Display the plot (maybe after saving).
    plt.show()



#! Function: Analyze data from a dataframe.
def analyze_data(data: pd.DataFrame, write_to_file: bool = False) -> None:
    """
    Analyze and then print statistics of data from a dataframe.
    The reviews are limited to 200 characters for better readability.
    The full data statistics are also written to a file called
    'data_statistics_<time>.txt' (optional).

    Args:
        data (pd.DataFrame): dataframe containing the data.

    Example:
    - Print statistics of data from a dataframe.
        Data statistics:
            Number of reviews: 25000
            Number of positive reviews: 12500
            Number of negative reviews: 12500
            Positive/Negative ratio: 1.0

        Positive reviews:
            I like this film very much!

        Negative reviews:
            I hate this film very much!

       (then write the statistics to a file called 'data_statistics.txt' if needed)
    """
    # Get positive and negative reviews.
    positive_reviews = data[data["sentiment"] == "pos"]
    negative_reviews = data[data["sentiment"] == "neg"]

    # Randomly sample one of the positive and negative reviews.
    reviews_samples = (
        positive_reviews["review"].sample(1).values[0],
        negative_reviews["review"].sample(1).values[0],
    )

    # Adjust the samples for better readability.
    ## Add newlines after each 70 characters.
    reviews_samples_adjusted = [
        re.sub(r"(.{70})", r"\1\n", review) for review in reviews_samples
    ]
    ## Max width of each sample is 200 characters.
    reviews_samples_adjusted = [
        review[:200] + "\n... (tldr)" if len(review) > 200 else review
        for review in reviews_samples_adjusted
    ]
    ## Add a tab before each line.
    reviews_samples_adjusted = [
        re.sub(r"^\s*", "\t", review, flags=re.MULTILINE)
        for review in reviews_samples_adjusted
    ]

    # Print statistics.
    print(
        f"""Data statistics:
        Number of reviews: {data.shape[0]}
        Number of positive reviews: {positive_reviews.shape[0]}
        Number of negative reviews: {negative_reviews.shape[0]}
        Positive/Negative ratio: {positive_reviews.shape[0] / negative_reviews.shape[0]}
        """
    )

    # Print positive and negative reviews.
    print("Positive reviews:")
    print(reviews_samples_adjusted[0])
    print()

    print("Negative reviews:")
    print(reviews_samples_adjusted[1])
    print()

    # Write the statistics and reviews to a file called 'data_statistics_<time>.txt'.
    if write_to_file:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        with open(f"data_statistics_{current_time}.txt", "w", encoding="utf-8") as f:
            f.write(
                f"""Data statistics:
                Number of reviews: {data.shape[0]}
                Number of positive reviews: {positive_reviews.shape[0]}
                Number of negative reviews: {negative_reviews.shape[0]}
                Positive/Negative ratio: {positive_reviews.shape[0] / negative_reviews.shape[0]}
                """
            )
            f.write("\n\n")
            f.write("Positive reviews:\n")
            f.write("\t" + reviews_samples[0])
            f.write("\n\n")
            f.write("Negative reviews:\n")
            f.write("\t" + reviews_samples[1])

    # Print a line to separate the output.
    print("-" * 100)
