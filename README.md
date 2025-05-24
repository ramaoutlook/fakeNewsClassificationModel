# Fake News Detection

This project implements a machine learning model to detect fake news. It utilizes various classification algorithms to differentiate between genuine and fake news articles based on their text content.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Manual Testing](#manual-testing)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

## Project Overview

The goal of this project is to build a reliable system for identifying fake news. The process involves:

1.  Loading and combining fake and genuine news datasets.
2.  Preprocessing the text data by cleaning and transforming it into a numerical format.
3.  Training multiple classification models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) on the processed data.
4.  Evaluating the performance of each model.
5.  Providing a manual testing function to predict the authenticity of new news articles.

## Dataset

The project uses two CSV files: `True.csv` containing genuine news articles and `Fake.csv` containing fake news articles. These datasets were combined and labeled accordingly.

## Data Preprocessing

The following steps were performed to prepare the text data for model training:

1.  **Combining Datasets:** The `True.csv` and `Fake.csv` datasets were concatenated into a single DataFrame, with a 'label' column indicating whether a news article is fake (0) or genuine (1).
2.  **Dropping Columns:** Irrelevant columns such as 'title', 'subject', and 'date' were removed.
3.  **Shuffling Data:** The combined dataset was shuffled to ensure a random distribution of fake and genuine news.
4.  **Text Cleaning:** A `wordopt` function was defined and applied to the 'text' column to perform the following cleaning operations:
    -   Convert text to lowercase.
    -   Remove URLs, HTML tags, punctuation, digits, and newline characters.
5.  **Sampling:** A subset of the data (500 rows for each class) was selected for faster processing during model training.
6.  **Vectorization:** The cleaned text data was transformed into numerical feature vectors using `TfidfVectorizer`. This process converts text into a matrix where each row represents a document and each column represents a unique word, with values indicating the importance of the word in the document.

## Model Training

The following classification models were trained on the preprocessed and vectorized data:

-   Logistic Regression
-   Decision Tree Classifier
-   Random Forest Classifier
-   Gradient Boosting Classifier

The data was split into training and testing sets (70% training, 30% testing) using `train_test_split`.

## Model Evaluation

The performance of each trained model was evaluated using the following metrics:

-   **Accuracy Score:** The proportion of correctly classified instances.
-   **Classification Report:** Provides precision, recall, F1-score, and support for each class.
-   **Confusion Matrix:** A table summarizing the performance of a classification algorithm.

The accuracy and classification reports for each model were printed to assess their effectiveness.

## Manual Testing

A `manual_testing` function was created to allow users to input a news article and receive predictions from all trained models. The input text is preprocessed using the same `wordopt` function and then vectorized using the fitted `TfidfVectorizer` before being fed into the models for prediction.

## Getting Started

To run this project, you will need to:

1.  Clone the repository to your local machine.
2.  Install the required dependencies (see [Dependencies](#dependencies)).
3.  Ensure you have the `True.csv` and `Fake.csv` datasets in the same directory as your notebook or script.
4.  Run the code cells in the notebook sequentially.

## Dependencies

The project requires the following Python libraries:

-   `numpy`
-   `pandas`
-   `re`
-   `nltk` (with stopwords downloaded)
-   `sklearn`

You can install these dependencies using pip:
