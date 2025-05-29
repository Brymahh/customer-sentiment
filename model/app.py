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
    '''
    Receives input from the user either through text area or file upload.
    '''
    # Text input area or file upload
    text_input_method = st.radio("Choose input method:", ("Direct Text Input", "Upload Text File"))

    if text_input_method == "Direct Text Input":
        user_input = st.text_area("Text Box", value="", height=200, placeholder="Enter your text here.")
    else:
        uploaded_file = st.file_uploader("Upload a Text File (.txt)", type=["txt"])
        if uploaded_file is not None:
            user_input = uploaded_file.read().decode("utf-8")
        else:
            user_input = ""

    return user_input


def predict(input):
    '''
    This function takes the input from the user, preprocesses it, and predicts the sentiment.
    '''    
    with open('model/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf=pickle.load(f)

    with open('model/nb_model.pkl', 'rb') as f:
        model=pickle.load(f)
    
    vectorized_input =tfidf.transform(input)
    pred = model.predict(vectorized_input)

    if pred[0]==1:
        #st.write('This is a Postive Feedback')
        st.write(f'This has a {model.predict_proba(vectorized_input)[0][1]*100:.2f}% Positive Feedback')
    else:
        #st.write('This is a Negative feedback')
        st.write(f'This has a {model.predict_proba(vectorized_input)[0][0]*100:.2f}% Negative Feedback')
    #st.write(f'The first probability is {model.predict_proba(vectorized_input)[0][0]}')
    #st.write(f'The second probability is {model.predict_proba(vectorized_input)[0][1]}')


def sentiment_response(final_input):
    '''
    This function handles the sentiment response based on the user's input.
    '''
    if not final_input.strip():
        st.warning("Kindly enter some text or upload a file to analyze sentiment.")
    else:
        final_input = pd.Series([final_input])
        st.success("Sentiment analysis completed successfully!")
        predict(final_input)



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
    #final_input = pd.Series([clean_input])
    
    if st.button('Find Sentiment'):
        sentiment_response(clean_input)


if __name__ == '__main__':
    main()
