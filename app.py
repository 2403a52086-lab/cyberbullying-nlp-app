import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("Cyberbullying Detection App")
st.write("Enter a comment below to check if it contains cyberbullying.")

user_input = st.text_area("Enter Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = preprocess(user_input)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("⚠️ Cyberbullying Detected")
        else:
            st.success("✅ Normal Comment")


