import sys
import pickle
import nltk
import re

nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


import pandas as pd
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
# connect to the database
    engine = create_engine('sqlite:///' + database_filepath)


# run a query
    df = pd.read_sql_query('SELECT * FROM DisasterData', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text.
    Args:
        text: text string
    Returns:
        (str[]): array of clean tokens
    """
    #Convert to lowercase and remove puncuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed

#     lemmatizer = WordNetLemmatizer()
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#     return clean_tokens


def build_model():
    """Builds classification model """
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
    ('tfidf', TfidfTransformer(use_idf = True)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10, min_samples_split = 10)))
    ])
                             
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_split':[2, 5, 10]}
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 10)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
 
    """
    Evaluate the model against a test dataset
    Args:
        model: Trained model
        X_test: Test features
        Y_test: Test labels
        category_names: String array of category names
    """
    y_preds = model.predict(X_test)
    print(classification_report(y_preds, Y_test.values, target_names=category_names))
    print("**** Accuracy scores for each category *****\n")
    for i in range(36):
        print("Accuracy score for " + Y_test.columns[i], accuracy_score(Y_test.values[:,i],y_preds[:,i]))


def save_model(model, model_filepath):
    
    """
    Save the model to a Python pickle
    Args:
        model: Trained model
        model_filepath: Path where to save the model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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