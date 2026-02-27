import streamlit as st
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Cyberbullying Detection App")
st.write("Enter a comment below to check if it contains cyberbullying.")

user_input = st.text_area("Enter Text Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("⚠️ Cyberbullying Detected")
        else:
            st.success("✅ Normal Comment")



