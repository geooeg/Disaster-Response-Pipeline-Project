import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_data(database_filepath):
    '''
    # load data from database
    :param database_filepath: the path of the database
    :return: training dataset, training data labels dataset, training data labels names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    Y = df.iloc[:,5:]

    return X, Y, Y.keys()


def tokenize(text):
    '''
    tokenize the text
    :param text: text to be tokenized
    :return: tokens of the text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build model using pipeline and GridSearch
    :return: a GridSearchCV instance of the estimator
    '''
    # Build a machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])


    # specify parameters for grid search
    parameters = {
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
        'clf__estimator__bootstrap': [True, False]
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate the model. Report the f1 score, precision and recall for each output category of the dataset
    :param model:
    :param X_test: test data
    :param Y_test: test data lebels
    :param category_names: name of labels
    :return: none
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=Y_test.keys()))

    pass


def save_model(model, model_filepath):
    '''
    save model as a pickle file
    :param model: model to be saved
    :param model_filepath: path to save the model
    :return: none
    '''

    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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