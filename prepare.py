from sklearn.model_selection import train_test_split
import pandas as pd

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
    
    return df
    
def prep_telco(df, train_size, seed):
    #1. Split the data
    train, test, val = split_data(df, train_size, seed)
    
    #2. Drop columns
    train = drop_columns(train)
    test = drop_columns(test)
    val = drop_columns(val)
    
    #3. Fix Data Types
    

    return train, test, val


def encode_contract_types(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded contract types 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.contract_type)
    train["contract_type_encoded"] = encoder.transform(train.contract_type)
    test["contract_type_encoded"] = encoder.transform(test.contract_type)
    validate["contract_type_encoded"] = encoder.transform(validate.contract_type)
    return encoder, train, test, validate