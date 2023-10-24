import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns


# for using TensorFlow for deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# for using PyTorch for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Load your symptom-disease data
data = pd.read_csv("Symptom2Disease.csv")

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Apply TF-IDF vectorization to the preprocessed text data
X = tfidf_vectorizer.fit_transform(data['text'])

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size=0.2, random_state=42)

# Initialize the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Set Streamlit app title with emojis
st.title("Healthcare Symptom-to-Disease Recommender 🏥👨‍⚕️")

# Define a sidebar
st.sidebar.title("Tool Definition")
st.sidebar.markdown("This tool helps you identify possible diseases based on the symptoms you provide. It is not a substitute for professional medical advice. Always consult a healthcare professional for accurate diagnosis and treatment.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to preprocess user input
def preprocess_input(user_input):
    user_input = user_input.lower()  # Convert to lowercase
    user_input = re.sub(r"[^a-zA-Z\s]", "", user_input)  # Remove special characters and numbers
    user_input = " ".join(user_input.split())  # Remove extra spaces
    return user_input

# Function to predict diseases based on user input
def predict_diseases(user_clean_text):
    user_input_vector = tfidf_vectorizer.transform([user_clean_text])  # Vectorize the cleaned user input
    predictions = model.predict(user_input_vector)  # Make predictions using the trained model
    return predictions

# Add user input section
user_input = st.text_area("Enter your symptoms (how you feel):", key="user_input")

# Add button to predict disease
if st.button("Predict Disease"):
    # Display loading message
    with st.spinner("Diagnosing patient..."):
        # Check if user input is not empty
        if user_input:
            cleaned_input = preprocess_input(user_input)
            predicted_diseases = predict_diseases(cleaned_input)

            # Display predicted diseases
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": f"Based on your symptoms, you might have {', '.join(predicted_diseases)}."})

            st.write("Based on your symptoms, you might have:")
            for disease in predicted_diseases:
                st.write(f"- {disease}")
        else:
            st.warning("Please enter your symptoms before predicting.")

# Display a warning message
st.warning("Please note that this tool is for informational purposes only. Always consult a healthcare professional for accurate medical advice.")

# Add attribution
st.markdown("Created with ❤️ by Your Name")
