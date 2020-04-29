def auto_payment(train, test, validate):
    for df in [train, test, validate]:
        df['automatic_payment'] = df.payment_type.str.contains('automatic').astype(int)
    return train, test, validate

def has_internet(train, test, validate):
    for df in [train, test, validate]:
            df['has_internet'] = ~ df.streaming_tv.str.contains('internet').astype(int)
    return train, test, validate


def create_features(df):
    df['has_internet'] = ~ df.streaming_tv.str.contains('internet').astype(int)
    df['automatic_payment'] = df.payment_type.str.contains('automatic').astype(int)
    
    return df