import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# --- Load data and model ---
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

@st.cache_resource
def load_sentiment_model():
    return load_model("rnn_model.h5")

@st.cache_data
def get_word_mappings():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

model = load_sentiment_model()
word_index, reverse_word_index = get_word_mappings()

# --- Helper functions ---
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# --- Streamlit UI ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Analyze the **sentiment** of any movie review — is it positive or negative?")

st.divider()

user_input = st.text_area("✍️ Enter your movie review below:", height=150, placeholder="Type or paste your review here...")

col1, col2 = st.columns([1, 2])
with col1:
    classify_btn = st.button("🔍 Classify Review")

if classify_btn:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a movie review before classifying.")
    else:
        with st.spinner("Analyzing sentiment..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "☹️ Negative"
            confidence = float(prediction[0][0])

        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Confidence:** `{confidence:.4f}`")

        st.progress(confidence if confidence <= 1 else 1.0)

        if confidence > 0.75:
            st.info("The model is quite confident about this prediction.")
        elif confidence < 0.25:
            st.info("The model seems uncertain. Try rephrasing the review.")
else:
    st.info("👆 Enter a review and click *Classify Review* to get started.")
