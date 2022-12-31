import sys
import argparse
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads datasets from 2 filepaths.
    
    Parameters:
    messages_filepath: data about messages from csv file
    categories_filepath: data about categories from csv file
    
    Returns:
    df: dataframe was merged from messages_filepath and categories_filepath 
    
    """
    #input data form file:
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge the data:
    df = messages.merge(categories, how='inner', on=['id'])
    return df
   


def clean_data(df):
     """
    Cleans data of dataframe.
    
    input:
    df: DataFrame
    
    ouput:
    df: Cleaned DataFrame
    
    """
    # make a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    #select the needed row and modify it
    row = categories.iloc[0]
    category_colnames = row.str.slice(stop=-2)
    categories.columns = category_colnames
    for col in categories:
        # set each character to be the last person of the string
        categories[col] = categories[col].astype(str).str[-1]
        # change type of data to int (suitable for calculate)
        categories[col] = categories[col].astype(int)
        # change all values not in (0,1) to 1
        categories.loc[categories[col] > 1, col] = 1
    # drop the first classes column from 'df'
    df = df.drop(['categories'], axis = 1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner")
    df = df.drop_duplicates()
    return df
   


def save_data(df, database_filepath):
    """save df in a SQLite database."""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('disaster_messages', con = engine, index=False, if_exists='replace')
    

def main():
    """Loads data, cleans data, saves data to database"""
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
