import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge two datasets using the common 'id' and assign to df
    df = messages.merge(categories, left_on = 'id', right_on = 'id', how = 'outer') 
    return df

def clean_data(df):
    #transform the categories column
    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0]
    category_colnames = list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames

    #create dummies for categories
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1] #get 0 or 1 of string
        categories[column] = pd.to_numeric(categories[column]) #string to numeric

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, join = 'inner')

    #remove duplicates
    # check number of duplicates
    print('Number of duplicates' ,df.duplicated().sum())
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    print(df.duplicated().sum())

    df = df.drop(df[df.related == 2].index)
    return df

def save_data(df, database_filename):
    database_filename = 'sqlite:///'+database_filename
    engine = create_engine(database_filename)
    df.to_sql('disasterResponseTbl', engine, index = False, if_exists = 'replace')

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