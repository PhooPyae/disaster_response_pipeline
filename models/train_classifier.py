from sqlalchemy import create_engine
import pandas as pd
import pickle
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import sys


def load_data(database_filepath):
    database_filepath = 'sqlite:///'+ database_filepath
    engine = create_engine(database_filepath)
    #read table
    df = pd.read_sql_table('disasterResponseTbl', con = engine) 
    X = df.message.values
    Y = df.drop(['id','message','original','genre'], axis = 1).values
    categories = df.drop(['id','message','original','genre'], axis = 1)
    return X, Y, categories

def tokenize(text):
    '''convert text into to word tokens, lemmatize and remove stopwords'''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_model():
    #build the model pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    #hyperparameters tuning with GridSearch
    parameters = {
        'clf__estimator__n_estimators': [10, 15, 20],
        'clf__estimator__min_samples_split': [2, 3, 4]

    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    #print the precision, recall and f1-score of each categories
    for i in range(36):
        print(category_names[i])
        print(classification_report(Y_test[:,i],y_pred[:,i]))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

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