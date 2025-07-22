import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.stem import SnowballStemmer

# Load dataset
df = pd.read_csv("twitter_data.csv")  # Adjust path if needed
df = df[['tweet', 'class']].dropna()

# Basic custom stopword list
basic_stopwords = set("""
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he he'd he'll he's her here here's hers herself him himself his how
    how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
    mustn't my myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some such than that
    that's the their theirs them themselves then there there's these they they'd they'll
    they're they've this those through to too under until up very was wasn't we we'd
    we'll we're we've were weren't what what's when when's where where's which while who
    who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves
""".split())

stemmer = SnowballStemmer("english")

# Clean function
def clean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return " ".join([stemmer.stem(word) for word in text.split() if word not in basic_stopwords])

# Clean tweets
df['tweet'] = df['tweet'].apply(clean)

# Features and labels
X = df['tweet']
y = df['class']  # 0 = hate speech, 1 = offensive language, 2 = neither

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vect = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict a test input
test_text = "I will kill you"
test_cleaned = clean(test_text)
test_vector = vectorizer.transform([test_cleaned])
predicted_class = model.predict(test_vector)[0]

label_map = {
    0: "Hate Speech Detected",
    1: "Offensive Language Detected",
    2: "No Hate or Offensive Language"
}

print(f"Prediction for '{test_text}': {label_map[predicted_class]}")