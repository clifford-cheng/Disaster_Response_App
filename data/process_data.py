import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    #Load and read messages csv
    messages = pd.read_csv(messages_filepath)
    
    #Load and read categories csv
    categories = pd.read_csv(categories_filepath)
    
    #Merge messages and categories csv
    df = pd.merge(messages, categories, left_index = True, on = ['id'])
    return df



def clean_data(df):
    #Create a dataframe of 36 individual category columns. 
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    #Take the first row of categories. Use copy to preserve original categories data. 
    row = categories.iloc[0]
    
    #Clean data. Remove numeric at end. Outcome is list of 36 category column names. 
    
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    
    #Rename columns with list created earlier. 
    categories.columns = category_colnames
    
    #Clean df and keep last character (1 or 0).
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Drop categories column and merge numeric data. Rename all columns. 
    # Drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    
    # Remove rows with a related value of 2 from the dataset
    df = df[df['related'] != 2]
    
    return df

def save_data(df, database_filename):
    """
    Saves clean dataset into an sqlite database
    Args:
        df:  Cleaned dataframe
        database_filename: Name of the database file
    """
    
    
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///..//data/DisasterResponse.db')
    df.to_sql('DisasterData', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()