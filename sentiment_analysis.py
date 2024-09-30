# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords
nltk.download('stopwords')

# Load the data
positive_reviews = []
negative_reviews = []

with open("C:/Users/ch.Tharani/Desktop/assignment/rt-polaritydata/rt-polarity.pos", "r", encoding='latin-1') as pos_file:
    positive_reviews = pos_file.readlines()

with open("C:/Users/ch.Tharani/Desktop/assignment/rt-polaritydata/rt-polarity.neg", "r", encoding='latin-1') as neg_file:
    negative_reviews = neg_file.readlines()


# Create labels: 1 for positive, 0 for negative
positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

# Combine the data and labels
all_reviews = positive_reviews + negative_reviews
all_labels = positive_labels + negative_labels

# Data Preprocessing function
def preprocess_text(text):
    # Lowercase, remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to all reviews
all_reviews_cleaned = [preprocess_text(review) for review in all_reviews]

# Splitting the data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(all_reviews_cleaned, all_labels, train_size=8000, stratify=all_labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=1000, stratify=y_temp, random_state=42)

# Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_tfidf, y_train)

# Predicting on the test data
y_pred = svm.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Display Results
print("Confusion Matrix:")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
