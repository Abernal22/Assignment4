from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

# Load and clean dataset
reviews = load_files('aclImdb/train', categories=['pos', 'neg'], encoding='utf-8')
X, y = reviews.data, reviews.target

#convert to tf-idf
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_tfidf = vectorizer.fit_transform(X)

# train testand split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)


with open('data.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
