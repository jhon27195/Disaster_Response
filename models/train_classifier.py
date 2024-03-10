import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  f1_score,precision_score,recall_score,accuracy_score,make_scorer, classification_report
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
nltk.download('wordnet') 
nltk.download('stopwords')
nltk.download('punkt')
import pickle

def load_data(database_filepath):
    """
    Loads disaster data from SQLite database
    
    Inputs:
        database_filepath(str): path to the SQLite database
        table_name(str): name of the table where the data is stored
        
    Returns:
        (DataFrame) X: Independent Variables , array which contains the text messages
        (DataFrame) y: Dependent Variables , array which contains the labels to the messages
        (DataFrame) category: Data Column Labels , a list the category names
    """
    
    
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM df", engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category = y.columns
    return X, y, category
   

def tokenize(text):
    """
        Tokenizes message data
        Input:
           text (string): message text data
        Output:
            (DataFrame) clean_messages: array of tokenized message data
    """ 
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words_stemmed = [PorterStemmer().stem(w) for w in words]
    words_Lem = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words_stemmed ]

    return words

    


def build_model():
    pipeline = Pipeline([
                    ('count_vectorizer', CountVectorizer(tokenizer = tokenize)),
                    ('tfidf_transformer', TfidfTransformer()),
                    ('MultiOutput_Random_Forest_Classifier', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    parameters = {
              'MultiOutput_Random_Forest_Classifier__estimator__n_estimators':[10],
              'MultiOutput_Random_Forest_Classifier__estimator__min_samples_split':[2]}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    model=cv
    return model

def evaluate_model(model, X_test, y_test, category):
    """
    Evaluates models performance in predicting message categories
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category)
    print(class_report)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()