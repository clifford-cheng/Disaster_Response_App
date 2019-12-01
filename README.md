# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
- process_data.py: The ETL pipeline used to process data in preparation for model building. This code takes csv data containing messages and categories, cleans it, and creates an SQLite database.

- train_classifier.py: This code takes the SQLite database to train and tune a ML model for categorizing the messages. The output is a pickle file where the fitted model resides. If the model takes too long to build, change n_estimators to a lower number. 



