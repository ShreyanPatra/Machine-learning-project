import streamlit as st
import sklearn
import pickle
import string
import nltk

nltk.download('punkt')
from nltk.corpus import stopwords
import pandas as pd
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

pt = PorterStemmer()


def process(text):
    text = text.lower()
    # tokenize
    text = nltk.word_tokenize(text)

    # remove special character
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    # remove stopwords and  punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

            # stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(pt.stem(i))
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title('Email spamm classifier')
input_sms = st.text_input('Enter the mesage')
if st.button('Predict'):

    transformed_sms = process(input_sms)

    vector_sms = tfidf.transform([transformed_sms])

    result = model.predict(vector_sms)[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("Not spam")