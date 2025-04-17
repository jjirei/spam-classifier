# spam_classifier_train.py

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample training data
texts = ["Free entry in 2 a wkly comp", "Hey, are we still meeting later?", "WINNER! You've won a prize!"]
labels = [1, 0, 1]  # 1 = spam, 0 = ham

# Train vectorizer and model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Save them as proper .pkl files
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
