# Hate_speech_detection

This is a machine learning project that detects hate speech and offensive language in tweets using Natural Language Processing (NLP) and a Random Forest Classifier.
It classifies input text into:
- Hate Speech  
- Offensive Language  
- Neither

# Features

1. Cleans and preprocesses raw tweets  
2. Uses stemming and custom stopword removal  
3. TF-IDF vectorization of text  
4. Trained using Random Forest Classifier  
5. Evaluates accuracy and shows confusion matrix  
6. Allows testing on custom input (e.g., "I will kill you")

# Dataset

- The dataset is to be named twitter_data.csv
- It must contain at least two columns:
   tweet - the tweet content
   class - labels (0 = hate speech, 1 = offensive language, 2 = neither)

# Requirements

Install the required packages using:
pandas
numpy
scikit-learn
nltk
