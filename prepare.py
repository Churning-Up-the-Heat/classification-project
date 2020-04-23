from sklearn.model_selection import train_test_split

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
    telco[telco.total_charges.str.contains(' ')].
    
def prep_telco(df):
    #1. Split the data
    train, test, val = split_data(df)
    
    #2. Drop columns
    train = drop_columns(train)
    test = drop_columns(test)
    val = drop_columns(val)

    return train, test, val