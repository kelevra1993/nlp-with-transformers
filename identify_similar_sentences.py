import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

train_csv_file = "/Users/Robert/PycharmProjects/nlp-with-transformers/data/small_training_dataframe.csv"
valid_csv_file = "/Users/Robert/PycharmProjects/nlp-with-transformers/data/validation_dataframe.csv"
test_csv_file = "/Users/Robert/PycharmProjects/nlp-with-transformers/data/test_dataframe.csv"

# Load the data
training_dataframe = pd.read_csv(train_csv_file)["text"]
validation_dataframe = pd.read_csv(valid_csv_file)["text"]
test_dataframe = pd.read_csv(test_csv_file)["text"]

# Convert the dataframes to lists
training_sentences = list(training_dataframe)
validation_sentences = list(validation_dataframe)
test_sentences = list(test_dataframe)

# Combine the datasets for vectorization
combined_sentences = training_sentences + validation_sentences + test_sentences

# Vectorize the sentences using TF-IDF
vectorizer = TfidfVectorizer().fit(combined_sentences)

# Transform sentences to TF-IDF vectors
train_vectors = vectorizer.transform(training_sentences)
valid_vectors = vectorizer.transform(validation_sentences)
test_vectors = vectorizer.transform(test_sentences)

# Calculate cosine similarity between training and validation sets
cosine_sim_train_valid = cosine_similarity(train_vectors, valid_vectors)

# Calculate cosine similarity between training and test sets
cosine_sim_train_test = cosine_similarity(train_vectors, test_vectors)

# Define a similarity threshold to consider sentences as near-duplicates
threshold = 0.60


# Function to check if two sentences should be considered near-duplicates
def is_near_duplicate(sent1, sent2, length_ratio_threshold=0.70):
    len1, len2 = len(sent1.split()), len(sent2.split())
    length_ratio = min(len1, len2) / max(len1, len2)
    return length_ratio > length_ratio_threshold


# Find near-duplicates between training and validation
duplicates_train_valid = []
for i in range(len(training_sentences)):
    for j in range(len(validation_sentences)):
        if cosine_sim_train_valid[i, j] > threshold and is_near_duplicate(training_sentences[i],
                                                                          validation_sentences[j]):
            duplicates_train_valid.append(
                (training_sentences[i], validation_sentences[j], np.round(cosine_sim_train_valid[i, j], 2)))

# Find near-duplicates between training and test
duplicates_train_test = []
for i in range(len(training_sentences)):
    for j in range(len(test_sentences)):
        if cosine_sim_train_test[i, j] > threshold and is_near_duplicate(training_sentences[i], test_sentences[j]):
            duplicates_train_test.append(
                (training_sentences[i], test_sentences[j], np.round(cosine_sim_train_test[i, j], 2)))

print(f"Found {len(duplicates_train_valid)} near-duplicates between training and validation.")
print(f"Found {len(duplicates_train_test)} near-duplicates between training and test.")

# Optionally, print out the near-duplicates
print("\nNear-duplicates between training and validation:")
for train, valid, score in duplicates_train_valid:
    print(f"Training: {train}\nValidation: {valid}\nScore:{score}\n")

print("\nNear-duplicates between training and test:")
for train, test, score in duplicates_train_test:
    print(f"Training: {train}\nTest: {test}\nScore:{score}\n")
