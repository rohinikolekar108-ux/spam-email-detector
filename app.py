import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📧 Spam Email Classifier")

# Load dataset
data = pd.read_csv("spam.csv")
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X = data['message']
y = data['label']

# Vectorization
vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# User input
st.subheader("Enter Email Text:")
email_input = st.text_area("")

# Prediction
if st.button("Check Email"):
    email_vec = vectorizer.transform([email_input])
    prediction = model.predict(email_vec)

    if prediction[0] == 1:
        st.error("🚨 This is a SPAM Email")
    else:
        st.success("✅ This is NOT a Spam Email")
