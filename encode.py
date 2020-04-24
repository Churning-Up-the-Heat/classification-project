from env import user, password, host

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

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

def encode_internet_service_types(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded internet service types 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.internet_service_type)
    train["encoded_internet_service_type"] = encoder.transform(train.internet_service_type)
    test["encoded__service_type"] = encoder.transform(test.internet_service_type)
    validate["encoded__service_type"] = encoder.transform(validate.internet_service_type)
    return encoder, train, test, validate

def encode_payment_types(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded payment types 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.payment_type)
    train["payment_type_encoded"] = encoder.transform(train.payment_type)
    test["payment_type_encoded"] = encoder.transform(test.payment_type)
    validate["payment_type_encoded"] = encoder.transform(validate.payment_type)
    return encoder, train, test, validate

def encode_churn(train, test, validate):
    '''Takes in train, test and validate dataframes
    Returns each df with a new coloumn for encoded churn 
    as well as the encoder used'''
    encoder = LabelEncoder()
    encoder.fit(train.churn)
    train["churn_encoded"] = encoder.transform(train.churn)
    test["churn_encoded"] = encoder.transform(test.churn)
    validate["churn_encoded"] = encoder.transform(validate.churn)
    return encoder, train, test, validate


def encoded_df(train, test, validate):
    '''takes in train, test, and validate df and encoded the following:
    contract_types
    internet service types
    churn
    payment types
    returns encoders for each and train, test, validate
    '''
    encoder_1, train, test, validate = encode_contract_types(train, test, validate)
    encoder_2, train, test, validate = encode_internet_service_types(train, test, validate)
    encoder_3, train, test, validate = encode_churn(train, test, validate)
    encoder_4, train, test, validate = encode_payment_types(train, test, validate)
    return encoder_1, encoder_2, encoder_3, encoder_4, train, test, validate