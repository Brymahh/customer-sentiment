import pandas as pd
import numpy as np
import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from main import preprocess

# 1. Recieve input from a user
# 2. Predict sentiment


def users_input():
        # Text input area or file upload
    text_input_method = st.radio("Choose input method:", ("Direct Text Input", "Upload Text File"))

    if text_input_method == "Direct Text Input":
        user_input = st.text_area("Paste Text", value="", height=200, help="Enter your text here.")
    else:
        uploaded_file = st.file_uploader("Upload a Text File (.txt)", type=["txt"])
        if uploaded_file is not None:
            user_input = uploaded_file.read().decode("utf-8")
        else:
            user_input = "" 

    return user_input


def predict(input):
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf=pickle.load(f)

    with open('model/nb_model.pkl', 'rb') as f:
        model=pickle.load(f)
  
    # tfidf=files['tfidf_vectorizer']
    # model = files['model']
    
    vectorized_input =tfidf.transform(input)
    pred = model.predict(vectorized_input)

    if pred[0]==1:
        #st.write('This is a Postive Feedback')
        st.write(f'This is a {model.predict_proba(vectorized_input)[0][1]} Positive Feedback')
    else:
        #st.write('This is a Negative feedback')
        st.write(f'This is a {model.predict_proba(vectorized_input)[0][0]} Negative Feedback')
    #st.write(f'The first probability is {model.predict_proba(vectorized_input)[0][0]}')
    #st.write(f'The second probability is {model.predict_proba(vectorized_input)[0][1]}')




def main():
    st.set_page_config(
    page_title="Sentiment",
    layout="wide",
    initial_sidebar_state="expanded"
    )

    with st.container():
        st.title('Sentiment Input Test')
        st.write("We'd absolutely love your input, \n" 
            "\nKindly provide us with your comments or reservations towards our services")
    
    input = users_input()
    clean_input=preprocess(input)
    
    if st.button('Find Sentiment'):
        predict(clean_input)


if __name__ == '__main__':
    main()
