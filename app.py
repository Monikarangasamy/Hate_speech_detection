# app.py

import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Load model and vectorizer
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess input
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# App UI
st.title("Hate Speech Detection")
user_input = st.text_area("Enter a tweet or text to analyze")

if st.button("Classify"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]

    if result == 0:
        st.error("Hate Speech")
    elif result == 1:
        st.warning("Offensive Language")
    else:
        st.success("Neutral / Neither")
