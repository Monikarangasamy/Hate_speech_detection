# hate_speech_model.py

import pandas as pd
import numpy as np
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("twitter_data.csv")

# Preprocess tweet text
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)                  # remove URLs
    text = re.sub(r"@\w+|#", "", text)                   # remove mentions and hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_tweet'] = df['tweet'].apply(clean_text)

# Features and labels
X = df['clean_tweet']
y = df['class']  # 0: hate speech, 1: offensive, 2: neither

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'rf_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
