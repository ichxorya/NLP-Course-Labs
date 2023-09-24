"""
This file contains utility functions for training and evaluating machine learning models.
"""

#! Import libraries.
import os
import re
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Import scikit-learn libraries.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#! Function: Train and evaluate a model.
def train_and_evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vectorizer_type: str,
    classifier_type: str,
    write_to_console: bool = True,
) -> (pd.DataFrame, dict):
    """
    Train and evaluate a model.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The test data.
        vectorizer_type (str): The type of vectorizer to use.
        classifier_type (str): The type of classifier to use.
        write_to_console (bool): Whether to print the evaluation results to the console.

    Returns:
        (pd.DataFrame, dict): The trained model and the classification report.
    """
    # Start counting the runtime.
    start_time = time.time()

    # Vectorize the text using the specified vectorizer type.
    vectorizer_type = vectorizer_type.lower()

    if vectorizer_type == "bow":
        vectorizer = CountVectorizer()
    elif vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer()
    elif vectorizer_type == "unigram":
        vectorizer = CountVectorizer(ngram_range=(1, 1))
    elif vectorizer_type == "bigram":
        vectorizer = CountVectorizer(ngram_range=(2, 2))
    elif vectorizer_type == "trigram":
        vectorizer = CountVectorizer(ngram_range=(3, 3))
    elif vectorizer_type == "unigram-tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
    elif vectorizer_type == "bigram-tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    elif vectorizer_type == "trigram-tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(3, 3))
    else:
        raise ValueError("Unsupported vectorizer type: {}".format(vectorizer_type))

    X_train = vectorizer.fit_transform(train_df["review"])
    X_test = vectorizer.transform(test_df["review"])
    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    # Train the specified classifier type.
    classifier_type = classifier_type.lower()
    max_iter = 10000

    if classifier_type == "naivebayes" or classifier_type == "nb":
        classifier = MultinomialNB()
    elif classifier_type == "logisticregression" or classifier_type == "lr":
        classifier = LogisticRegression(max_iter=max_iter)
    elif classifier_type == "supportvectormachine" or classifier_type == "svm":
        classifier = sk.svm.LinearSVC(max_iter=max_iter)
    else:
        raise ValueError("Unsupported classifier type: {}".format(classifier_type))

    classifier.fit(X_train, y_train)

    # Predict the test set.
    y_pred = classifier.predict(X_test)

    # Start the evaluation.
    print(
        "Model evaluation for {} classifier with {} vectorizer:".format(
            classifier_type.upper(), vectorizer_type.upper()
        )
    )

    ## Evaluate the model.
    accuracy = accuracy_score(y_test, y_pred)

    ## Get classification report.
    model_classification_report = classification_report(y_test, y_pred, output_dict=True)

    # Print the evaluation results to the console.
    if write_to_console:
        # Print the runtime.
        print("\t" + "Runtime: {:.2f} seconds".format(time.time() - start_time))

        # Print the accuracy and classification report.
        print("\t" + "Accuracy: {:.2f}%".format(accuracy * 100))
        print(classification_report(y_test, y_pred))

        # Print a line to separate the output.
        print("-" * 100)

    # Return the trained model and the classification report.
    return (classifier, model_classification_report)
