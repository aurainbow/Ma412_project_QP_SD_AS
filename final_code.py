#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class: Ma412 - Mathematical Foundations for Statistical Learning
Project: Multi-Label Classification of Scientific Literature Using the NASA SciX Corpus
Authors: Quentin Pomes, and Aurane Schaff 
"""


# ------ Building a Multi-Label Classification Model ------


# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# ------ Importing files ------


# Loading dataset containing first five categories
# Importing necessary files from folder (parquet files)
train_file = "/Users/quentin/Desktop/IPSA/A4/projet_math_A4/train-00000-of-00001-b21313e511aa601a.parquet"
val_file = "/Users/quentin/Desktop/IPSA/A4/projet_math_A4/val-00000-of-00001-66ce8665444026dc.parquet"

# Load and preprocess the dataset, we juste have to replace between the two to use the file needed
data = pd.read_parquet(val_file) # 3025 sets
# data = pd.read_parquet(train_file) # 18677 sets


# ------ Preprocessing the data ------


# Define the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    # Combine the 'title' and 'abstract' columns, replacing missing values with an empty string
    return ' '.join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word.isalpha() and word not in stop_words
    )

# Apply the preprocessing function to clean the text data
# Combine the 'title' and 'abstract' columns, replacing missing values with an empty string
# Allows for better feature extraction
data['combined'] = (data['title'].fillna('') + ' ' + data['abstract'].fillna('')).apply(preprocess_text)

# Define the input features X and target labels y
X = data['combined']
y = data['verified_uat_labels']

# Binarize the multi-label targets using MultiLabelBinarizer
# It converts the multi-label targets into a binary matrix format suitable for the classifier
mlb = MultiLabelBinarizer() 
y_binarized = mlb.fit_transform(y)

# Split the data into training and testing sets
# - 80% of the data is used for training
# - 20% of the data is used for testing
# This ensures that the model is trained on a subset of the data and tested on the remaining data 
# it prevents overfitting as it is evaluated on unseen data

# Split into train and test, it's an example
X_train, X_test, y_train, y_test = train_test_split(X, y_binarized, test_size=0.2, random_state=42)

# Define parameters for the SGDClassifier
sgd_params = dict(alpha=1e-5, penalty="l2", loss="log_loss")

# Define parameters for the CountVectorizer
vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)


# ------ Supervised Pipeline ------


# Streamline preprocessing and model training into a single workflow
pipeline = Pipeline([
    ("vect", CountVectorizer(**vectorizer_params)),
    ("tfidf", TfidfTransformer()),
    ("clf", OneVsRestClassifier(SGDClassifier(**sgd_params))),
])

# CountVectorizer : it converts the text into a matrix of token counts, that's represent the frequency of word or phrases 
# TfidfTransformer : it transforms the count matrix into a Term Frequency-Inverse Document Frequency (tfidf) representation, it normalizes the token frequency
# SGDClassifier : it creates the model for predicting labels based on the processed features


# ------ Supervised Learning ------


# Here we are training the supervised model on the entire labeled dataset and evaluate
print("Supervised SGDClassifier on 100% of the data:")

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Predict labels for the test set (evaluate the model)
y_pred_supervised = pipeline.predict(X_test)

# Calculate the micro-averaged F1 score, accuracy, and confusion matrix
f1_supervised = f1_score(y_test, y_pred_supervised, average="micro")

# Accuracy is the number of correct predictions divided by the total number of predictions
accuracy_supervised = accuracy_score(y_test, y_pred_supervised)

# Confusion matrix is a table that is often used to describe the performance of a classification model
# Calculated using argmax for multi-label data
confusion_supervised = confusion_matrix(y_test.argmax(axis=1), y_pred_supervised.argmax(axis=1))

# Print the results
print("Micro-averaged F1 score (Supervised) on test set: {:.3f}".format(f1_supervised))
print("Accuracy (Supervised) on test set: {:.3f}".format(accuracy_supervised))
print("Confusion Matrix (Supervised):\n", confusion_supervised)

# Pseudo-Labeling Function
def pseudo_labeling_with_partial_labels(percentage, threshold=0.5):
    print(f"Pseudo-Labeling with {percentage}% of the training data (rest is pseudo-labeled):")
    
    # Randomly select a subset of the training data as labeled based on the given percentage
    mask = np.random.rand(len(y_train)) < (percentage / 100)

    # Subset of labeled data
    X_labeled, y_labeled = X_train[mask], y_train[mask]

    # Remaining data treated as unlabeled
    X_unlabeled = X_train[~mask]
    
    # Train on Labeled Data
    # Create a pipeline for vectorization, TF-IDF transformation, and classification
    pipeline = Pipeline([
        ("vect", CountVectorizer(**vectorizer_params)),
        ("tfidf", TfidfTransformer()),
        ("clf", OneVsRestClassifier(SGDClassifier(**sgd_params))),
    ])

    # With that we can train the pipeline on the labeled data
    pipeline.fit(X_labeled, y_labeled)
    
    # Predict Pseudo-Labels for Unlabeled Data
    # Predict probabilities for each label on the unlabeled data
    # Convert probabilities to binary labels based on the threshold
    y_unlabeled_probs = pipeline.predict_proba(X_unlabeled)
    y_unlabeled_pseudo = (y_unlabeled_probs >= threshold).astype(int)
    
    # Combine the original labeled data with the pseudo-labeled data
    X_combined = np.concatenate((X_labeled, X_unlabeled))
    y_combined = np.concatenate((y_labeled, y_unlabeled_pseudo))
    
    # Retrain the pipeline on the combined dataset
    pipeline.fit(X_combined, y_combined)
    
    # Predict labels for the test set
    # Calculate the micro-averaged F1 score, accuracy, and confusion matrix
    y_pred = pipeline.predict(X_test)
    f1_pseudo = f1_score(y_test, y_pred, average="micro")
    accuracy_pseudo = accuracy_score(y_test, y_pred)
    confusion_pseudo = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    
    print(f"Micro-averaged F1 score (Pseudo-Labeling) on test set: {f1_pseudo:.3f}")
    print(f"Accuracy (Pseudo-Labeling) on test set: {accuracy_pseudo:.3f}")
    print("Confusion Matrix (Pseudo-Labeling):\n", confusion_pseudo)

# We are going to use 20%, 40%, 60% and 80% to analyze how the amount of labeled data affects performance
for pct in [20, 40, 60, 80]:
    pseudo_labeling_with_partial_labels(pct)

# Print results for 100%
print("\nResults for 100% labeled data:")
print(f"Micro-averaged F1 score: {f1_supervised:.3f}")
print(f"Accuracy: {accuracy_supervised:.3f}")
print("Confusion Matrix:\n", confusion_supervised)