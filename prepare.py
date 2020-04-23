import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def split_data(df, train_size, seed):
    # Create the train and test sets
    train, test = train_test_split(df, train_size=train_size, random_state=seed)
    # Create the validate set by splitting from train set
    train, validate = train_test_split(train, train_size=train_size, random_state=seed)
    
    return train, test, validate

def drop_columns(df):
    # Dropping all of the columns used just to merge the data together in the SQL query. These columns will provide nothing good for our models or exploration
    return df.drop(columns=['payment_type_id',
                            'internet_service_type_id',
                            'contract_type_id'])

def fix_dtypes(df):
    # Setting new customers total charges to zero so we can convert the dtype from object to float
    df.total_charges = df.total_charges.str.replace(' ', '0')
    df.total_charges = df.total_charges.astype(float)
    
    return df

def prep_telco(df, train_size, seed):
    #1. Split the data
    train, test, validate = split_data(df, train_size, seed)
    
    #2. Drop columns
    train    = drop_columns(train)
    test     = drop_columns(test)
    validate = drop_columns(validate)
    
    #3. Fix Data Types
    train    = fix_dtypes(train)
    test     = fix_dtypes(test)
    validate = fix_dtypes(validate)

    return train, test, validate


