#!/bin/python
# this script will be used to preprocess the whole data
# attention that to avoid data leakage, the test data will be normalized 
# using the train data statistics. be careful to not use the test data statistics
# in the training process
import pandas as pd

def calculate_target_variable(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """ The goal of the model is to predict the direction of the stock price.
    Because of this, the target variable will be a binary variable that indicates
    if the stock price will go up or down. This function will calculate the target
    variable based on the stock price."""
    df['target'] = (df[column_name] - df[column_name].shift(1)) > 0

def calculate_returns(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """ Calculate the returns of the stock price. """
    df['returns'] = df[column_name].pct_change()

def read_data():
    """ Read the parquet files from the data/raw_combined folder and return a generator 
    to iterate over the data. """
    pass

# def get_average