# sentimental-analysis
Overview
This project implements a binary sentiment classifier for movie reviews using a Support Vector Machine (SVM) model. The classifier is trained on a dataset containing positive and negative reviews to predict sentiment polarity.

Dataset
The dataset used for this project is the RT Polarity Dataset, which consists of:
5,331 positive sentences in rt-polarity.pos
5,331 negative sentences in rt-polarity.neg
Data Splitting
The dataset is split into three sets:

Training Set: The first 4,000 positive and 4,000 negative reviews.
Validation Set: The next 500 positive and 500 negative reviews.
Test Set: The final 831 positive and 831 negative reviews.


Libraries Used
The following libraries are used in this project:
numpy
pandas
sklearn (scikit-learn)
nltk (Natural Language Toolkit)
matplotlib
seaborn

You can install the necessary libraries using the following command:
pip install numpy pandas scikit-learn nltk matplotlib seaborn

Implementation
1. Data Preprocessing
The data preprocessing steps include:
Loading positive and negative reviews.
Lowercasing text, removing punctuation, and filtering out stopwords.
2. Model Training
An SVM classifier with a linear kernel is trained on the TF-IDF representation of the training data.

3. Model Evaluation
The model is evaluated using the test set, and the following metrics are reported:
True Positives (TP)
True Negatives (TN)
False Positives (FP)
False Negatives (FN)
Precision
Recall
F1 Score
The confusion matrix is visualized to assess the model's performance.

Confusion Matrix
A confusion matrix is generated to provide a visual representation of the model's predictions:
True Negatives (TN): 621
False Positives (FP): 210
False Negatives (FN): 212
True Positives (TP): 619

Running the Code
To run the sentiment analysis, execute the following command:
python sentiment_analysis.py
Make sure to place the rt-polaritydata folder in the same directory as the script or update the paths accordingly in the code.

Conclusion
This project demonstrates a basic implementation of sentiment analysis using machine learning techniques. The SVM classifier effectively distinguishes between positive and negative movie reviews based on the training data.
