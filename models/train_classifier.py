import sys
import re
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """
    Loads data from database.
    
    Input:
    database_filepath: Filepath to the database
    
    Output:
    X: Features
    Y: Target
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', con=engine)
    X = df['message']
    Y = df.iloc[:, 5:]
    return X,Y



def tokenize(text):
    """
    Tokenize and lemmatize text.
    
    Input:
    text: Text to be tokenized
    
    Output:
    clean_tokens: cleaned tokens 
    """
        # Convert to lowercase
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize text
    tokens = text.split()

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # Reduce words to their lemma
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return tokens


def build_model():
    """
    Build classifier and tune model.
    
    output:
    cv: Classifier 
    """  
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {'clf__estimator__n_estimators' : [50, 100]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv

def display_results(y_test, y_pred):
    confusion_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\tConfusion Matrix:\n", confusion_mat)
    print("\tReport:\n", report)

    
def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the performance of model and return classification report. 
    
    Input:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Output:
    Classification report for each column
    """
    y_pred = model.predict(X_test)

    # best hyper-parameters
    print("\nBest Parameters:", model.best_params_)

    for i, col in enumerate(Y_test):
        print(col, classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    """ output is the final model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))

def load_model(model_filepath):
    return pickle.load(open(model_filepath, 'rb'))


def main():
    """ Build the model, train the model, evaluate the model, save the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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
