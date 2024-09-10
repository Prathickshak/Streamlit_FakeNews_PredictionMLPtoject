import numpy as np
import pandas as pd
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_model():
    with open('fake_news_saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
port_stem = data['port_stem']
vectorizer = data['vectorizer']
stop_words = set(stopwords.words('english'))

def predict_page_show():

    st.title("Welcome to Fake News Prediction")

    title = st.text_input("Enter the title of the news: ")
    author = st.text_input("Enter the author of the news: ")
    text = st.text_input("Enter the text of the news: ")

    ok = st.button("Predict")

    if ok:
        user_input = {
            "title": [title],
            "author": [author],
            "text": [text]
        }
        user_input = pd.DataFrame(user_input)

        # Combine author and title into a new column 'content'
        user_input['content'] = user_input['author'] + ' ' + user_input['title']

        # Stemming function applied to each row of the 'content' column
        def stemming(content):
            stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
            stemmed_content = stemmed_content.lower()
            stemmed_content = stemmed_content.split()
            stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
            stemmed_content = ' '.join(stemmed_content)
            return stemmed_content

        # Apply stemming to each string in the 'content' column
        user_input['content'] = user_input['content'].apply(stemming)

        # Transform the content using the vectorizer
        x = vectorizer.transform(user_input['content'])

        # Make the prediction
        prediction = model.predict(x)

        # Display the prediction result
        st.write(f"The prediction is: {prediction[0]}")

        if (prediction[0]==0):
            st.success("The news is real :)")
        else:
            st.warning("The news is fake :(")