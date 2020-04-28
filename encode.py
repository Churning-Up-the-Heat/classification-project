from env import user, password, host

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

def create_encoded_dfs(train, test, validate):

    cols = ['contract_type', 'internet_service_type', 'churn',
            'payment_type', 'online_security', 'tech_support',
            'device_protection', 'online_backup', 'paperless_billing',
            'partner', 'dependents', 'internet_service_type', 
            'streaming_tv', 'streaming_movies']
    
    encoded_train    = train.copy()
    encoded_test     = test.copy()
    encoded_validate = validate.copy()
    
    for col in cols:
        for df in [train, test, validate]:
            df[col].str.replace('No internet service', '0')
            df[col].str.replace('No phone service', '0')
        encoder = LabelEncoder()
        encoder.fit(encoded_train[col])
        
        encoded_train[col]    = encoder.transform(encoded_train[col])
        encoded_test[col]     = encoder.transform(encoded_test[col])
        encoded_validate[col] = encoder.transform(encoded_validate[col])
        
    return encoded_train, encoded_test, encoded_validate

def encode_contract_types(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded contract types 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.contract_type)
    train["contract_type_encoded"] = encoder.transform(train.contract_type)
    test["contract_type_encoded"] = encoder.transform(test.contract_type)
    validate["contract_type_encoded"] = encoder.transform(validate.contract_type)
    return train, test, validate

def encode_internet_service_types(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded internet service types 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.internet_service_type)
    train["encoded_internet_service_type"] = encoder.transform(train.internet_service_type)
    test["encoded__service_type"] = encoder.transform(test.internet_service_type)
    validate["encoded__service_type"] = encoder.transform(validate.internet_service_type)
    return train, test, validate

def encode_payment_types(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded payment types 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.payment_type)
    train["payment_type_encoded"] = encoder.transform(train.payment_type)
    test["payment_type_encoded"] = encoder.transform(test.payment_type)
    validate["payment_type_encoded"] = encoder.transform(validate.payment_type)
    return train, test, validate

def encode_churn(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded churn 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.churn)
    train["churn_encoded"] = encoder.transform(train.churn)
    test["churn_encoded"] = encoder.transform(test.churn)
    validate["churn_encoded"] = encoder.transform(validate.churn)
    return train, test, validate

def encode_online_security(train, test, validate):
    encoder = LabelEncoder()
    encoder.fit(train.online_security)
    train["online_security_encoded"] = encoder.transform(train.online_security)
    test["online_security_encoded"] = encoder.transform(test.online_security)
    validate["online_security_encoded"] = encoder.transform(validate.online_security)
    return train, test, validate

def encode_tech_support(train, test, validate):
    encoder = LabelEncoder()
    encoder.fit(train.tech_support)
    train["tech_support_encoded"] = encoder.transform(train.tech_support)
    test["tech_support_encoded"] = encoder.transform(test.tech_support)
    validate["tech_support_encoded"] = encoder.transform(validate.tech_support)
    return train, test, validate

def encode_device_protection(train, test, validate):
    encoder = LabelEncoder()
    encoder.fit(train.device_protection)
    train["device_protection_encoded"] = encoder.transform(train.device_protection)
    test["device_protection_encoded"] = encoder.transform(test.device_protection)
    validate["device_protection_encoded"] = encoder.transform(validate.device_protection)
    return train, test, validate

def encode_online_backup(train, test, validate):
    encoder = LabelEncoder()
    encoder.fit(train.online_backup)
    train["online_backup_encoded"] = encoder.transform(train.online_backup)
    test["online_backup_encoded"] = encoder.transform(test.online_backup)
    validate["online_backup_encoded"] = encoder.transform(validate.online_backup)
    return train, test, validate


def encoded_df(train, test, validate):
    '''takes in train, test, and validate df and encoded the following:
        contract_types
        internet service types
        churn
        payment types
    returns train, test, validate
    '''
    train, test, validate = encode_contract_types(train, test, validate)
    train, test, validate = encode_internet_service_types(train, test, validate)
    train, test, validate = encode_churn(train, test, validate)
    train, test, validate = encode_payment_types(train, test, validate)
    train, test, validate = encode_online_security(train, test, validate)
    train, test, validate = encode_tech_support(train, test, validate)
    train, test, validate = encode_device_protection(train, test, validate)
    train, test, validate = encode_online_backup(train, test, validate)
    
    return train, test, validate

