import acquire
import prepare
import features
import encode

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

def run_model():
    # Get the Data
    df = acquire.get_telco_data()

    # Prepare the Data
    df = prepare.drop_columns(df)
    df = prepare.fix_dtypes(df)

    # Add Features
    df = features.create_features(df)

    # Encode DataFrame
    df = encode.encode_df(df)

    # Select features to be used in the model
    cols = ['contract_type', 
            'tenure',
            'monthly_charges',
            'payment_type',
            'has_internet']

    X = df[cols]
    y = df.churn
    
    # Create and fit the model
    forest = RandomForestClassifier(n_estimators=100, 
                                      max_depth=9,
                                      random_state=123).fit(X, y)

    # Create a DataFrame to hold predictions
    results = pd.DataFrame(
        {'Costumer_ID': df.customer_id,
         'Model_Predictions': forest.predict(X),
         'Model_Probabilities': forest.predict_proba(X)[:,1]
        })

    # Generate csv
    results.to_csv('model_results.csv')

    return results
