import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')
#from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 1. Recieve the users input
# 2. Preprocess (Tokenize the input)
# 3. Perform Sentiment Analysis
# 4. Return the results


# load data
def load_data():
    '''
    This function loads the file and cleans the target variable g
    '''
    data = pd.read_csv('dataset/ecommerce_reviews.csv')

    # map to positive and neative
    map_values= {
    '__label__1': 0, 
    '__label__2': 1
    }
    data['labels'] = data['labels'].map(map_values)
    X = data['text']
    y = data['labels']
    
    return X, y



#clean & tokenize
def preprocess(text):
    """" 
    Takes sentence as input, tokenizes and eliminates stopwords
    """
    #remove numbers and symbols
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #remove urls
    text = re.sub(r'http\S+', '', text)

    #remove stopwords
    stop_words= stopwords.words('english')
    word_tokens = nltk.word_tokenize(text)
    filtered = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_text = " ".join(filtered) 

    return filtered_text



#train and save the model
def model_vectorize(text,y):

    #split data
    X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer()
    X_train_vect = tfidf.fit_transform(X_train)
    X_test_vect = tfidf.transform(X_test)


    # modeling
    mb = MultinomialNB()
    mb.fit(X_train_vect,y_train)
    y_pred = mb.predict(X_test_vect)
    acc = accuracy_score(y_test,y_pred)
    cls_report = classification_report(y_test, y_pred)
    print(f'accuracy: {acc}')
    print(f'Classification Report: {cls_report}')

    return tfidf, mb
   


def main():

    # Load the data
    X, y = load_data()

    # Preprocess the data
    text = X.apply(preprocess)

    # Vectorize and train the model
    tfidf, model = model_vectorize(text, y) 

    # saved_steps = {'tfidf_vectorizer': tfidf, 'model': model}
    # with open('model/saved_steps.pkl', 'wb') as f:
    #     pickle.dump(saved_steps,f)

    with open("model/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open("model/nb_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # resolve iterable error
    # text2 = "This is a great product! I love it."
    # text2=[text2]
    # # using the model
    # vectorized_input2 = tfidf.transform(text2)
    # pred = model.predict(vectorized_input2)
    # print(f'The prediction for the text "{text2}" is: {pred[0]}')

if __name__ == '__main__':
    main()



