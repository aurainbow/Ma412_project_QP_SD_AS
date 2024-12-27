"""
Class: Ma412 - Mathematical Foundations for Statistical Learning
Project: Multi-Label Classification of Scientific Literature Using the NASA SciX Corpus
Authors: Quentin Pomes, and Aurane Schaff 
"""


# ------ Class distribution Study ------


# Importing the necessary libraries
# pip install pandas pyarrow
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from itertools import combinations

# Importing necessary files from folder (parquet files)
train_file = "/Users/aurane/Downloads/year_4/S1/Ma412(maths_avancees)/project/train.parquet"

# read the parquet file
train = pd.read_parquet(train_file)

# Place verified_uat_labels into a list
verified_uat_labels = train['verified_uat_labels'].tolist()

# function to flatten a list (create unique list from nested list)
def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

# Flatten the verified_uat_labels list 
labels_w_duplicates = flatten(verified_uat_labels)


# ------ Count the frequency of each label ------


# Count the frequency of each label
label_freq = Counter(labels_w_duplicates)

# Sort the labels by frequency in descending order
sorted_label_freq = dict(sorted(label_freq.items(), key=lambda x: x[1], reverse=True))

# randomize the order of labels and their frequencies
import random
random.seed(42)
random_labels = random.sample(sorted_label_freq.keys(), len(sorted_label_freq))
random_label_freq = {label: sorted_label_freq[label] for label in random_labels}

# Plot the distribution of all the labels in order of frequency
plt.figure(figsize=(10, 6))
plt.bar(range(len(sorted_label_freq)), list(sorted_label_freq.values()))
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.title("Distribution of All Labels in Order of Frequency")
plt.show()

# Plot the distribution of all the labels in random order
plt.figure(figsize=(10, 6))
plt.bar(range(len(random_label_freq)), list(random_label_freq.values()))
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.title("Distribution of All Labels in Random Order")
plt.show()

# Print the total number of unique labels
print(f"Total number of unique labels: {len(sorted_label_freq)}")

# print the five most common labels
print(f"Five most common labels: {list(sorted_label_freq.items())[:5]}")

# print the five least common labels
print(f"Five least common labels: {list(sorted_label_freq.items())[-5:]}")


# ------ Count the frequency of each combination of labels ------


# count cooccurrences of labels
cooccurrences = Counter()

for labels in verified_uat_labels:
    for pair in combinations(labels, 2):
        cooccurrences[pair] += 1

# Sort the cooccurrences by frequency in descending order
sorted_cooccurrences = dict(sorted(cooccurrences.items(), key=lambda x: x[1], reverse=True))

# print the total number of unique label pairs
print(f"Total number of unique label pairs: {len(sorted_cooccurrences)}")

# print the five most common label pairs and their frequencies
print(f"Five most common label pairs: {list(sorted_cooccurrences.items())[:5]}")

# print the five least common label pairs and their frequencies
print(f"Five least common label pairs: {list(sorted_cooccurrences.items())[-5:]}")


# ------ Count the number of labels per document ------


# Count the number of labels per document
num_labels_per_doc = [len(labels) for labels in verified_uat_labels]

# Count the frequency of each number of labels per document
num_labels_freq = Counter(num_labels_per_doc)

# Sort the number of labels by frequency in descending order
sorted_num_labels_freq = dict(sorted(num_labels_freq.items(), key=lambda x: x[1], reverse=True))

# print the total number of unique number of labels per document
print(f"Total number of unique number of labels per document: {len(sorted_num_labels_freq)}")

# print the five most common number of labels per document and their frequencies
print(f"Five most common number of labels per document: {list(sorted_num_labels_freq.items())[:5]}")

# print the five least common number of labels per document and their frequencies
print(f"Five least common number of labels per document: {list(sorted_num_labels_freq.items())[-5:]}")