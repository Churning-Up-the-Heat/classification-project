import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def split_data(df, train_size=.8, seed=123):
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


def create_phone_lines_variable(train, test, validate):
    '''
    takes in train, test, validate dataframes and creates a new column that represents
    the information from phone_service and multiple_line in one variable 
    returns train, test, and validate df
    0 = No Phone Service
    1 = Single Phone Line
    2 = Multiple Lines
    '''
    #encode phone_service
    train["phone_service"] = train.phone_service.str.replace('No', "0")
    test["phone_service"] = test.phone_service.str.replace('No', "0")
    validate["phone_service"] = validate.phone_service.str.replace('No', "0")
    
    train["phone_service"] = train.phone_service.str.replace('0 phone service', "0")
    test["phone_service"] = test.phone_service.str.replace('0 phone service', "0")
    validate["phone_service"] = validate.phone_service.str.replace('0 phone service', "0")
    
    train["phone_service"] = train.phone_service.str.replace('Yes', "1")
    test["phone_service"] = test.phone_service.str.replace("Yes", "1")
    validate["phone_service"] = validate.phone_service.str.replace("Yes", "1")
    
    #convert phone_service to an integer
    train["phone_service"] = train.phone_service.astype("int")
    test["phone_service"] = test.phone_service.astype("int")
    validate["phone_service"] = validate.phone_service.astype("int")
    
    #encode multiple_lines
    train["multiple_lines"] = train.multiple_lines.str.replace('No', "0")
    test["multiple_lines"] = test.multiple_lines.str.replace('No', "0")
    validate["multiple_lines"] = validate.multiple_lines.str.replace('No', "0")
    
    train["multiple_lines"] = train.multiple_lines.str.replace('0 phone service', "0")
    test["multiple_lines"] = test.multiple_lines.str.replace('0 phone service', "0")
    validate["multiple_lines"] = validate.multiple_lines.str.replace('0 phone service', "0")
    
    train["multiple_lines"] = train.multiple_lines.str.replace('Yes', "1")
    test["multiple_lines"] = test.multiple_lines.str.replace('Yes', "1")
    validate["multiple_lines"] = validate.multiple_lines.str.replace('Yes', "1")
    
    #convert multiple_lines to an integer
    train["multiple_lines"] = train.multiple_lines.astype("int")
    test["multiple_lines"] = test.multiple_lines.astype("int")
    validate["multiple_lines"] = validate.multiple_lines.astype("int")
    
    # create new variable by summing the encoded
    train["phone_lines"] = train["phone_service"] + train["multiple_lines"]
    validate["phone_lines"] = validate["phone_service"] + validate["multiple_lines"]
    test["phone_lines"] = test["multiple_lines"] + test["phone_service"]
    
    #drop phone_service and mulitple_lines columns
    train.drop(columns=["multiple_lines", "phone_service"], inplace=True)
    test.drop(columns=["multiple_lines", "phone_service"], inplace=True)
    validate.drop(columns=["multiple_lines", "phone_service"], inplace=True)
    
    return train, test, validate


def dependent_partner_grouping(train, test, validate):
    '''
    takes in train, test, validate dataframes and creates a new column that represents
    the information from partner and dependents into one variable 
    returns train, test, and validate df
    0 = No partner or dependant
    1 = Either a dependent or partner
    2 = Both a dependent and partner
    '''
    
    #encode partner
    encoder = LabelEncoder()
    encoder.fit(train.partner)
    train["partner_encoded"] = encoder.transform(train.partner)
    test["partner_encoded"] = encoder.transform(test.partner)
    validate["partner_encoded"] = encoder.transform(validate.partner)
    
    #encode dependents
    encoder.fit(train.dependents)
    train["dependents_encoded"] = encoder.transform(train.dependents)
    test["dependents_encoded"] = encoder.transform(test.dependents)
    validate["dependents_encoded"] = encoder.transform(validate.dependents)
    
    # create new variable by summing the encoded
    train["dependent_partner_grouping"] = train["partner_encoded"] + train["dependents_encoded"]
    test["dependent_partner_grouping"] = test["partner_encoded"] + test["dependents_encoded"]
    validate["dependent_partner_grouping"] = validate["partner_encoded"] + validate["dependents_encoded"]
    
    #drop encoded partner and dependents column
    train.drop(columns=["partner_encoded", "dependents_encoded"], inplace=True)
    test.drop(columns=["partner_encoded", "dependents_encoded"], inplace=True)
    validate.drop(columns=["partner_encoded", "dependents_encoded"], inplace=True)
    
    return train, test, validate


def prep_telco(df, train_size=.8, seed=123):
    
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
    
    #4. Add tenure years to each
    train["tenure_years"] = (train.tenure / 12).round(1)
    test["tenure_years"] = (test.tenure / 12).round(1)
    validate["tenure_years"] = (validate.tenure / 12).round(1)
    
    #5 Add phone_lines features
    train, test, validate = create_phone_lines_variable(train, test, validate)
    
    # Add dependent_partner_grouping
    train, test, validate = dependent_partner_grouping(train, test, validate)

    return train, test, validate



